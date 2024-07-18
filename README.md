# Tree-Search-Algorithms
1. ChatGPT describes BFS as

    ```text
    Breadth-first search (BFS) is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root (or some arbitrary node of a graph, sometimes referred to as a 'search key') and explores the neighbor nodes first, before moving to the next level neighbours. It uses a queue data structure to keep track of the next node to visit. It is used to find the shortest path between two nodes in a graph.
    ```

    Summarize and then explain what ChatGPT gets right about BFS and what it misses out on.



   In summary, BFS is a search algorithm that visits all shallowest nodes or immediate neighbors from a current node at each stage of a search. It also uses a First in First out approach when visiting nodes.

   In addition, BFS is complete, meaning it will find a solution if one exists, and it is optimal for finding the shortest path in an unweighted graph. It can have high space complexity due to the need to store all nodes at      the current depth level in the queue. The iterations in the BFS algorithm are smooth while it explores breadthwise and find the nodes which are adjacently connected to a source node and there is no way for this method to      get stuck in an infinite loop.

2. I asked ChatGPT "Does it make sense to use BFS on a Binary Search Tree?" ... it responded with:

    ```text
    It can make sense to use Breadth-first search (BFS) on a Binary Search Tree (BST) in certain situations. One of those situations is when we need to traverse the entire tree level by level, for example, printing all the elements of the tree level by level. The output will be sorted since it is a binary search tree.

    Another example of using BFS on a BST is when we want to find the shortest path between two nodes. Since a BST has the property of being ordered, the shortest path between two nodes is the path that goes through the least number of levels. Since BFS visits the nodes level by level, it will find the shortest path between the two nodes.

    However, in most cases, it is more efficient to use the in-order, pre-order, or post-order traversal specific to BSTs, since they are designed to take advantage of the tree's ordered structure.
    ```

    Explain why ChatGPT is completely wrong.




   A traditinal Binary earch tree can be traversed through three basic algorithms: inorder, preorder, and postorder tree walks. Inorder tree walk: Nodes from the left subtree get visited first, followed by the root node and      right subtree. Such a traversal visits all the nodes in the order of non-decreasing key sequence. For Inorder, you traverse from the left subtree to the root then to the right subtree, for Preorder, you traverse from the      root to the left subtree then to the right subtree and for Post order, you traverse from the left subtree to the right subtree then to the root which in these cases BFS dows not work like that, BFS works by visiting nodes     breadthwise not depthwise.

   but in order to use BFS on a binary tree,we need to process the root node at level 0, and then we process the left and right child at level 1, visiting all the nodes from left to right at every level.

3. Explain when greedy searching can perform worse than uninformed search. Create a graph that demonstrates this and include the generated images below. _Hint_ you may need to create some fake data but in the same format that    you used to encode the Seattle data.




   Greedy search prioritise the immediate nodes with the least nodes first therefore it might not be the optimal as futher down the line there might be nodes with high cost that may increase overal cost to destination. For       example, consider a scenario where you are navigating a maze, and at each step, the greedy algorithm chooses the path that appears to be the shortest. However, this might lead the algorithm into a dead-end or a longer path    overall. In contrast, an uninformed search algorithm, such as breadth-first search, explores multiple paths systematically, ensuring that it considers all possibilities and eventually finds the optimal solution or depth       first search which might start off with the right node and head down straight to the goal with a least cost overall.


   
4. Try reversing directions and going from International District to Ballad. Do you get the same resulting paths for each of the algorithms? Explain why or why not and show the new images below.

   

   In reversing directions I did not get the same paths for BFS and DFS but i did get the same for A* search. Starting from International District, the BFS was able to visit other       shallow nodes closer to International District in addition to the other nodes on the path to Ballard, increasing the overall cost. that said, going from Ballrd to International       District had a lesser cost than this route. it is also the same for DFS when we use this direction except for A* search where the cost is the same and also the path is the same.

