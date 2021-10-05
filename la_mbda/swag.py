import tensorflow as tf
from tensorflow_addons.optimizers import SWA
from tensorflow_probability import stats as tfps


# Implements the following behavior:
# If the optimizer is warm, weights are exchanged with the shadow weights, sampled upon request
# and get swapped back to the original ones. Otherwise, do not swap and do not sample.
class WeightsSampler(object):
    def __init__(self, optimizer):
        self._opt = optimizer

    def __enter__(self):
        self._opt.swap_weights()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._opt.swap_weights()

    def sample(self, scale):
        self._opt.sample_and_assign(scale)


class SWAG(SWA):
    def __init__(self,
                 optimizer,
                 start_averaging=0,
                 average_period=10,
                 max_num_models=20,
                 decay=0.99,
                 **kwargs):
        verbose = kwargs.pop('verbose', True)
        super(SWAG, self).__init__(
            optimizer,
            start_averaging,
            average_period,
            "SWAG", **kwargs)
        self._set_hyper("max_num_models", max_num_models)
        self._set_hyper("decay", decay)
        self._iterations_count = tf.Variable(0, trainable=False)
        self._verbose = verbose

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        max_num_models = self._get_hyper("max_num_models", tf.int32)
        for var in var_list:
            self.add_slot(var, "mean")
            self.add_slot(var, "variance")
            numel = tf.size(var)
            self.add_slot(var, "cov_mat_sqrt", initializer=tf.zeros([max_num_models, numel]))
            self.add_slot(var, "copy")
        self._model_weight = var_list
        self._shadow_copy = [self.get_slot(var, "copy") for var in var_list]

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if "apply_state" in self._optimizer._dense_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad, var, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        mean_op, variance_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var, apply_state)
        return tf.group(train_op, mean_op, variance_op, cov_mat_sqrt_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        mean_op, variance_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var, apply_state)
        return tf.group(train_op, mean_op, variance_op, cov_mat_sqrt_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices
            )
        mean_op, variance_op, cov_mat_sqrt_op = self._apply_mean_op(train_op, var, apply_state)
        return tf.group(train_op, mean_op, variance_op, cov_mat_sqrt_op)

    def _apply_mean_op(self, train_op, var, apply_state):
        apply_state = apply_state or {}
        local_apply_state = apply_state.get((var.device, var.dtype.base_dtype))
        if local_apply_state is None:
            local_apply_state = self._fallback_apply_state(
                var.device, var.dtype.base_dtype
            )
        mean_var = self.get_slot(var, "mean")
        variance_var = self.get_slot(var, "variance")
        cov_mat_sqrt_var = self.get_slot(var, "cov_mat_sqrt")
        mean_op = self._mean_op(var, mean_var, variance_var, cov_mat_sqrt_var, local_apply_state)
        return mean_op

    @tf.function
    def swap_weights(self):
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        mean_period = self._get_hyper("average_period", tf.dtypes.int64)
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, mean_period),
        )
        max_num_models = self._get_hyper("max_num_models", tf.int64)
        if self.iterations >= start_averaging and num_snapshots >= max_num_models:
            for a_element, b_element in zip(self._shadow_copy, self._model_weight):
                a_identity = tf.identity(a_element)
                a_element.assign(b_element)
                b_element.assign(a_identity)

    @tf.function
    def _mean_op(self, var, mean_var, variance_var, cov_mat_sqrt_var, local_apply_state):
        mean_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, mean_period),
        )
        # The mean update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * mean_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            decay = self._get_hyper("decay", tf.float32)
            mean, variance = tfps.assign_moving_mean_variance(
                var, mean_var, variance_var,
                self._iterations_count, decay)
            cov_mat_sqrt_var_roll = tf.roll(cov_mat_sqrt_var, 1, 0)
            cov_mat_sqrt_var_roll = tf.concat([tf.reshape((var - mean), [1, -1]),
                                               cov_mat_sqrt_var_roll[1:]], 0)
            return mean, variance, cov_mat_sqrt_var.assign(
                cov_mat_sqrt_var_roll, use_locking=self._use_locking)
        return mean_var, variance_var, cov_mat_sqrt_var

    @tf.function
    def sample_and_assign(self, scale):
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        mean_period = self._get_hyper("average_period", tf.dtypes.int64)
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, mean_period),
        )
        max_num_models = self._get_hyper("max_num_models", tf.int64)
        if self.iterations >= start_averaging and num_snapshots >= max_num_models:
            for var in self._model_weight:
                var.assign(self.sample(scale, var), use_locking=self._use_locking)
        elif self._verbose:
            tf.print("SWAG is not warm yet.",
                     "\nIterations so far: ", self.iterations,
                     "\nWarmup time: ", start_averaging,
                     "\nSnapshots so far: ", num_snapshots,
                     "\nMax models: ", max_num_models)

    @tf.function(experimental_relax_shapes=True)
    def sample(self, scale, var):
        max_num_models = self._get_hyper("max_num_models", tf.float32)
        mean = self.get_slot(var, "mean")
        variance = self.get_slot(var, "variance")
        cov_mat_sqrt = self.get_slot(var, "cov_mat_sqrt")
        var_sample = tf.math.sqrt(variance / 2.0) * tf.random.normal(tf.shape(variance))
        cov_sample = tf.linalg.matmul(
            cov_mat_sqrt,
            tf.random.normal([tf.shape(cov_mat_sqrt)[0], 1]),
            transpose_a=True) / ((2.0 * (max_num_models - 1)) ** 0.5)
        rand_sample = var_sample + tf.reshape(cov_sample, tf.shape(var_sample))
        scale_sqrt = scale ** 0.5
        sample = (mean + scale_sqrt * rand_sample)
        return sample
