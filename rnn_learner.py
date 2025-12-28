import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as kb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Concatenate, ZeroPadding1D
import numpy as np
import pandas as pd
from util import DLogger

"""
This class implements an RNN learner which can be trained on a task to predict next actions.
Updated for TensorFlow 2.19: uses save_weights with `.weights.h5`, avoids mixed tensor/array errors.
"""

class RNNAgent:
    def __init__(self, n_actions, state_size, n_cells, reset_after=True, model_path=None):
        self.n_actions = n_actions
        self.n_cells = n_cells

        # Build model from scratch
        original_inputs = tf.keras.Input(shape=(None, 1 + n_actions + state_size), name='action_reward')
        initial_state = tf.keras.Input(shape=(n_cells,), name='initial_state')
        rnn_out = kl.GRU(n_cells, return_sequences=True, name='GRU', reset_after=reset_after)(
            original_inputs, initial_state=initial_state
        )
        policy = kl.Dense(n_actions, activation='softmax', name='policy')(rnn_out)
        self.model = tf.keras.Model(inputs=[original_inputs, initial_state], outputs=[rnn_out, policy], name='encoder')

        if model_path is not None:
            self.model.load_weights(model_path)
            DLogger.logger().debug('Weights loaded from ' + model_path)

        DLogger.logger().debug(f"Model created with {n_actions} actions and {n_cells} cells")

    def train(self, reward, action, state,
              test_reward=None, test_action=None, test_state=None,
              output_path=None, lr=1e-3):

        action, inputs = self._make_model_input(action, reward, state)
        DLogger.logger().debug('Training data dims: ' + str(action.shape))

        if test_action is not None:
            DLogger.logger().debug('Test data dims: ' + str(test_action.shape))

        test_inputs = None
        if test_reward is not None:
            test_action, test_inputs = self._make_model_input(test_action, test_reward, test_state)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        save_path = output_path + 'model-init.weights.h5'
        DLogger.logger().debug('Init model weights saved to: ' + save_path)
        self.model.save_weights(save_path)

        events = []

        try:
            for epoch in range(5000):
                if test_reward is not None and epoch % 100 == 0:
                    initial_state = tf.zeros((test_inputs.shape[0], self.n_cells), dtype=tf.float32)
                    _, test_pred_pol = self.model([test_inputs, initial_state])
                    test_actions_onehot_corrected = (
                        (1 - tf.reduce_sum(test_action, axis=2))[:, :, tf.newaxis] + test_action
                    )
                    test_loss = tf.reduce_sum(
                        kb.log(tf.reduce_sum(test_pred_pol * test_actions_onehot_corrected, axis=2)),
                        axis=[1]
                    )
                    mean_test_loss = -kb.sum(test_loss) / tf.reduce_sum(test_action)

                    correct_percent = kb.sum(
                        tf.cast(tf.reduce_sum(test_pred_pol * test_action, axis=2) > 1. / self.n_actions, tf.float32)
                    ) / tf.reduce_sum(test_action)

                    DLogger.logger().debug(
                        f'step {epoch}: mean test loss = {mean_test_loss.numpy()}, %correct = {correct_percent.numpy()}'
                    )

                    events.append({
                        'epoch': epoch,
                        'loss': mean_test_loss.numpy(),
                        'n actions': tf.reduce_sum(test_action).numpy(),
                        'sum loss': -kb.sum(test_loss).numpy(),
                        '% correct': correct_percent.numpy()
                    })

                    pd.DataFrame(events).to_csv(output_path + "events.csv")
                    save_path = output_path + f'model-{epoch}.weights.h5'
                    DLogger.logger().debug('Model weights saved to: ' + save_path)
                    self.model.save_weights(save_path)

                with tf.GradientTape() as tape:
                    initial_state = tf.zeros((inputs.shape[0], self.n_cells), dtype=tf.float32)
                    _, pred_pol = self.model([inputs, initial_state])
                    actions_onehot_corrected = (
                        (1 - tf.reduce_sum(action, axis=2))[:, :, tf.newaxis] + action
                    )
                    loss = tf.reduce_sum(
                        kb.log(tf.reduce_sum(pred_pol * actions_onehot_corrected, axis=2)),
                        axis=[1]
                    )
                    loss = -kb.sum(loss)

                    correct_percent = kb.sum(
                        tf.cast(tf.reduce_sum(pred_pol * action, axis=2) > 1. / self.n_actions, tf.float32)
                    ) / tf.reduce_sum(action)

                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                DLogger.logger().debug(
                    f'step {epoch}: mean loss = {loss.numpy():.3f}, correct = {correct_percent.numpy():.3f}'
                )

        except KeyboardInterrupt:
            DLogger.logger().debug('Training interrupted at trial ' + str(epoch))
            save_path = output_path + 'model-final.weights.h5'
            DLogger.logger().debug('Model weights saved to: ' + save_path)
            self.model.save_weights(save_path)

    def _make_model_input(self, action, reward, state):
        return self.make_model_input(action, reward, state, self.n_actions)

    @classmethod
    def make_model_input(cls, action, reward, state, n_actions):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        state = tf.convert_to_tensor(state, dtype=tf.float32) if state is not None else None

        action = tf.convert_to_tensor(to_categorical(action, num_classes=n_actions), dtype=tf.float32)
        action_reward = Concatenate(axis=2)([reward[:, :, tf.newaxis], action])
        action_reward = ZeroPadding1D(padding=(1, 0))(action_reward)

        if state is not None:
            inputs = Concatenate(axis=2)([action_reward[:, :-1, :], state])
        else:
            inputs = action_reward[:, :-1, :]

        return action, inputs
