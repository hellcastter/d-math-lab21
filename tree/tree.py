""" Decision Tree Classifier """
import heapq
import numpy as np

from sklearn.datasets import load_iris # (150, 4)
from sklearn.datasets import load_wine # (178 13)
from sklearn.datasets import load_breast_cancer # (569, 30)

from sklearn.model_selection import train_test_split

class Node:
    """ Node for a decision tree """
    def __init__(self, X: np.ndarray, y: np.ndarray, gini: float):
        self.X = X
        self.y = y

        self.gini = gini
        self.feature_index = 0
        self.threshold = 0.0

        self.left: Node | None = None
        self.right: Node | None = None

        self.class_number: int | None = None

    def detect_class(self):
        """ Detect to which class node is """
        self.class_number = np.bincount(self.y).argmax()


class DecisionTreeClassifier:
    """ Decision tree """
    def __init__(self, max_depth: int | None = None):
        """Decision tree

        Args:
            max_depth (int | None, optional): max height of the tree. 0 or None cancels max height.
            Defaults to None.
        """
        self.max_depth = max_depth
        self.root = None

    def gini(self, classes: np.ndarray) -> float:
        """A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem)

        Args:
            classes (np.ndarray): list of used classes

        Returns:
            float: gini index

        >>> Tree = DecisionTreeClassifier()
        >>> Tree.gini(np.array([1, 2, 3, 2, 1]))
        0.6399999999999999
        """
        gini_sum = 0
        number_of_classes = len(classes)

        for group_class in np.unique(classes, return_counts=True)[1]:
            gini_sum += (group_class / number_of_classes) ** 2

        return 1 - gini_sum

    def split_data(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        """Test all the possible splits.

        Args:
            X (np.ndarray): training data
            y (np.ndarray): training answers

        Returns:
            tuple[int, float, float]: index of feature, threshold and gini

        >>> Tree = DecisionTreeClassifier()
        >>> Tree.split_data(
        ...     np.array([[10, 7], [2, 10], [5, 7]]),
        ...     np.array([1, 0, 1])
        ... )
        (0, 3.5, 0.0)
        """
        number_of_features = len(X[0])
        number_of_classes = y.size

        index = 0
        threshold = 0.0
        lowest_gini = np.Inf

        # for all features
        for class_idx in range(number_of_features):
            # create heapq of feature column
            active_group = []

            for feature in X:
                element = feature[class_idx]
                heapq.heappush(active_group, element)

            # we have to get mean of two neighbor elements
            # 1 and 2, 2 and 3 and so on...
            # so we need len(active_group) - 1 iterations
            for _ in range(len(active_group) - 1):
                # mean of two smallest elements
                new_threshold = sum(heapq.nsmallest(2, active_group)) / 2
                heapq.heappop(active_group)

                # divide by left and right tree info
                under_threshold = X[:, class_idx] < new_threshold
                left_tree_y = y[under_threshold]
                right_tree_y = y[~under_threshold]

                # calc gini for children
                left_gini = self.gini(left_tree_y)
                right_gini = self.gini(right_tree_y)

                left_nodes_count = len(left_tree_y)

                # gini for this node
                # i/m * Gini_left + (m-i)/m * Gini_right
                gini = left_gini * (left_nodes_count / number_of_classes) +\
                    right_gini * (1 - (left_nodes_count / number_of_classes))

                if gini < lowest_gini:
                    lowest_gini = gini
                    index = class_idx
                    threshold = new_threshold

        return index, threshold, lowest_gini

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth=0) -> Node | None:
        """create a root node
        recursively split until max depth is not exceeded

        Args:
            X (np.ndarray): training data
            y (np.ndarray): training answers
            depth (int, optional): max depth of tree. Defaults to 0.

        Returns:
            Node | None: root node of decision tree
        """
        if self.max_depth and depth > self.max_depth:
            return None

        index, threshold, gini = self.split_data(X, y)

        if index is None:
            return None

        # current Node
        node = Node(X, y, gini)
        node.feature_index = index
        node.threshold = threshold
        node.detect_class()

        under_threshold = X[:, index] < threshold

        # left child data
        left_X = X[under_threshold]
        left_y = y[under_threshold]

        # right child data
        right_X = X[~under_threshold]
        right_y = y[~under_threshold]

        # can't divide on left and right
        if right_y.size == 0 or left_y.size == 0:
            return node

        node.left = self.build_tree(left_X, left_y, depth=depth + 1)
        node.right = self.build_tree(right_X, right_y, depth=depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        """basically wrapper for build tree / train

        Args:
            X (np.ndarray): training data
            y (np.ndarray): training answers
        """
        self.root = self.build_tree(X, y)

    def predict_one(self, test: np.ndarray) -> int | None:
        """Predict which class will test data have

        Args:
            test (np.ndarray): test data

        Returns:
            int: class index
        """
        root = self.root

        if root is None:
            print("Train your decision tree at first")
            return None

        while True:
            feature = root.feature_index

            if test[feature] < root.threshold:
                if root.left is None:
                    return root.class_number

                root = root.left
            else:
                if root.right is None:
                    return root.class_number

                root = root.right

    def predict(self, X_test: np.ndarray) -> list[int | None]:
        """traverse the tree while there is a child
        and return the predicted class for it, 
        note that X_test can be a single sample or a batch

        Args:
            X_test (np.ndarray): test data

        Returns:
            list[int | None]: list of classes
        """
        return [self.predict_one(test) for test in X_test]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """return accuracy

        Args:
            X_test (np.ndarray): test data
            y_test (np.ndarray): answers

        Returns:
            float: accuracy of prediction
        """
        return sum(self.predict(X_test) == y_test) / len(y_test)


if __name__ == "__main__":
    # Dataset. Choose one you like the most
    # X, y = load_iris(return_X_y=True) # (150, 4)
    X, y = load_wine(return_X_y=True) # (178 13)
    # X, y = load_breast_cancer(return_X_y=True) # (569, 30)

    print("Example of samples. Size:", X.shape)
    print(X[:2])

    print()
    print("Unique classes:", np.unique(y))

    # divide dataset to training and test data
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

    tree = DecisionTreeClassifier(10)
    tree.fit(X, y)

    print(f"Predicted classes: {tree.predict(X_test)}")
    print(f"Accuracy: {tree.evaluate(X_test, y_test) * 100}%")
