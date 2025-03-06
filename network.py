import numpy as np

class HopfieldNetwork:
    """
    A base Hopfield network class.
    """

    def __init__(self, hidden_size):
        """
        Initialize the Hopfield Network.

        Args:
            hidden_size: Number of neurons in the network.
        """

        self.hidden_size = hidden_size
        self.weights = np.zeros((hidden_size, hidden_size))
    

    def encode(self, patterns):
        """
        Train the network using the Hebbian learning rule.
        
        Args:
            patterns: List of patterns to store in the network (each pattern is a 1D numpy array of -1 and 1).
        """

        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)

        np.fill_diagonal(self.weights, 0) # no self-connections

    
    def recall(self, pattern, steps = 10):
        """
        Recall a pattern asynchronously by iterating multiple steps.

        Args:
            pattern: The initial pattern to start recall (1D numpy array of -1 and 1).
            steps: Number of steps to update neurons.
        
        Returns:
            state: The recalled pattern.
        """

        state = pattern.copy()
        for _ in range(steps):
            i = np.random.randint(0, self.hidden_size)  # choose a random neuron
            state[i] = np.sign(np.dot(self.weights[i], state))
            if state[i] == 0:
                state[i] = 1  # resolve zero case

        return state
    
    
    def energy(self, pattern):
        """
        Compute the energy of a given pattern.

        Args:
            pattern: A pattern to compute energy for.
        
        Returns:
            energy: Energy value (scalar).
        """

        energy = -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))
        return energy