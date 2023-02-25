""" Decision Tree Classifier """
import heapq
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split


class Node:
    """ Node for a decision tree """
    def __init__(self, X: pd.DataFrame, y: pd.Series, gini: float):
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
        self.class_number = self.y.mode()[0]


class DecisionTreeClassifier:
    """ Decision tree """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.root = None

    def gini(self, classes: pd.Series) -> float:
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem)

        >>> Tree.gini(pd.Series([1, 2, 3, 2, 1]))
        0.6399999999999999
        '''
        gini_sum = 0
        number_of_classes = classes.count()

        for group_class in classes.value_counts():
            gini_sum += (group_class / number_of_classes) ** 2

        return 1 - gini_sum


    def split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[int, float, float]:
        """
        Test all the possible splits in O(N*F) where N in number of samples
        and F is number of features
        return index and threshold value

        >>> Tree.split_data(
        ...     pd.DataFrame(np.array([[10, 7], [2, 10], [5, 7]]), columns=[0,1]),
        ...     pd.Series([1, 0, 1])
        ... )
        (0, 3.5, 0.0)
        """
        number_of_features = len(X.columns)
        number_of_classes = y.size

        index = 0
        threshold = 0.0
        lowest_gini = np.Inf

        # for all features
        for class_idx in range(number_of_features):
            # create heapq of feature column
            active_group = []

            for _, feature in X.iterrows():
                element = feature[class_idx]
                heapq.heappush(active_group, element)

            # we have to get mean of two neighbor elements
            # 1 and 2, 2 and 3 and so on...
            # so we need len(active_group) - 1 iterations
            for _ in range(1, len(active_group)):
                # mean of two smallest elements
                new_threshold = sum(heapq.nsmallest(2, active_group)) / 2
                heapq.heappop(active_group)

                # divide by left and right tree info
                left_tree_y = y[X[class_idx] < new_threshold]
                right_tree_y = y[X[class_idx] >= new_threshold]

                # calc gini for children
                left_gini = self.gini(left_tree_y)
                right_gini = self.gini(right_tree_y)

                left_nodes_count = left_tree_y.count()

                # gini for this node
                # i/m * Gini_left + (m-i)/m * Gini_right
                gini = left_gini * (left_nodes_count / number_of_classes) +\
                    right_gini * (1 - (left_nodes_count / number_of_classes))

                if gini < lowest_gini:
                    lowest_gini = gini
                    index = class_idx
                    threshold = new_threshold

        return index, threshold, lowest_gini


    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0) -> Node | None:
        """
        create a root node
        recursively split until max depth is not exceeded
        """
        if self.max_depth and depth > self.max_depth:
            return None

        index, threshold, gini = self.split_data(X, y)

        if index is None:
            return None

        node = Node(X, y, gini)
        node.feature_index = index
        node.threshold = threshold

        under_threshold = X[index] < threshold

        left_X = X[under_threshold]
        left_y = y[under_threshold]

        right_X = X[~under_threshold]
        right_y = y[~under_threshold]

        node.detect_class()

        if right_y.empty or left_y.empty:
            return node

        node.left = self.build_tree(left_X, left_y, depth=depth + 1)
        node.right = self.build_tree(right_X, right_y, depth=depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ basically wrapper for build tree / train """
        y = pd.Series(y)
        X = pd.DataFrame(X)

        self.root = self.build_tree(X, y)

    def predict_one(self, test: np.ndarray) -> int | None:
        """Predict which class will test data have

        Args:
            test (np.ndarray): income data

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
        """
        traverse the tree while there is a child
        and return the predicted class for it, 
        note that X_test can be a single sample or a batch
        """
        return [self.predict_one(test) for test in X_test]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """ return accuracy """
        return sum(self.predict(X_test) == y_test) / len(y_test)
# import doctest
# doctest.testmod(extraglobs={'Tree': DecisionTreeClassifier(0)})


# X, y = laod_iris(return_X_y=True)
# X, y = load_wine(return_X_y=True)
X, y = load_breast_cancer(return_X_y=True)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

tree = DecisionTreeClassifier(10)
tree.fit(X, y)

print(tree.evaluate(X_test, y_test))
