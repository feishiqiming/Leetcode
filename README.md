# Leetcode
## 704. Binary Search

### Abstract: 
### consider nums = [2], so while condition left <= right instead of left < right, also to avoid dead loop, right = mid -1, left = mid + 1.

tips: while condition left<=right because if left== right then this point remains unsearched.//keep searching when the search window still have numbers.
mid point location = left point location + distance->(right - left)//2    //prevent (left + right)/2 overflow
right = mid - 1; left = mid + 1// ensure search window gets shorter

special condition to be considered: if left == right and nums[mid] != target , if rewrite right = mid left = mid, then search window doesn't get shorter and becomes dead loop.

## 51. N-Queens
Python code:
```python

class Solution:
    """
    n皇后问题利用回溯法解决，本质上答案的寻找过程是深度优先搜索DFS-用递归方法更易懂，用迭代的话需要维护一个栈
    皇后放置位置规则中，我们可以在放置过程中放在不同行不同列以规避行列的规则,状态存储在columns中,columns下标i代表行,columns[i]代表列
    搜索过程中进行对角线规则判断
    """
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        def dfs(columns: List[int]):
            #对角线规则判断
            if not validate(columns):
                return 
            #递归结束基本条件-找到答案
            if len(columns)==n:
                temp = columns.copy()
                res.append(temp)
                return
            for i in range(n):
                if i not in columns:
                    columns.append(i)
                    dfs(columns)
                    columns.pop()
        
        def validate(columns: List[int]) -> bool:
            for i in range(len(columns)):
                for j in range(i+1,len(columns)):
                    #对角线规则判断
                    if columns[j] - columns[i] == j - i or columns[j] - columns[i] == i - j:
                        return False
            return True
        
        def transform(res: List[List[int]]):
            for i in range(len(res)):
                #n个.的字符串组成的n个元素的空棋盘数组
                board = ["."*n for _ in range(n)]
                for j in range(len(res[i])):
                    #棋盘的每一行填上对应列的皇后Q
                    board[j] = board[j][:res[i][j]]+'Q'+board[j][res[i][j] + 1 :]
                res[i] = board
                
        dfs([])
        transform(res)
        return res

```


## 155. Min Stack

### Abstract: 
### maintain 2 stacks, 1 is a normal stack, another is a stack containning the minimum value that far.


## 150. Evaluate Reverse Polish Notation

### Abstract: 
### Stack- If int then push,if operator then pop a number as B,then pop a number as A, push A 'operator' B



## 22. Generate Parentheses

### Abstract: 
### DFS using recursion

Python code:
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def dfs(temp:str,l:int,r:int,n:int):
            #reach unsuccessful nodes
            if(l>n or r>n or r>l):
                return
            #reach successful nodes
            if(l == n and r == n):
                res.append(temp)
                return
            dfs(temp+'(',l+1,r,n)
            dfs(temp+')',l,r+1,n)
        dfs('',0,0,n)
        return res    

```


## 200. Number of Islands

### Abstract: 
### DFS using recursion,bounding function-out of bounds, reach the water 0, reach the nodes visited previously.
### The main difference between graph traversal and tree traversal -> graph traversal may visit a same node twice.

Java code:
```Java
class Solution {
    public int numIslands(char[][] grid) {
        int res = 0;
        for(int i = 0;i<grid.length;i++){
           for(int j = 0;j<grid[0].length;j++){
               if(grid[i][j]=='1'){
                   dfs(grid,i,j);
                   res++;
               }
           } 
        }
        return res;

    }
    public void dfs(char[][] grid,int i,int j){
    
        //out of bounds
        if(j>=grid[0].length||i>=grid.length||i<0||j<0) return;
        
        //reach water or previously visited nodes
        if(grid[i][j] == '2'||grid[i][j] == '0'){
            return;
        }
        
        //mark the node as visited
        else if(grid[i][j] == '1') grid[i][j] = '2'; 
        
        dfs(grid,i,j+1);
        dfs(grid,i,j-1);
        dfs(grid,i+1,j);
        dfs(grid,i-1,j);
    }
}

```

## 79. Word Search

### Abstract: 
### Similar to Q200,DFS using recursion,bounding function-out of bounds, reach unmatched letter, reach the nodes visited previously.
### Before each search, we should restore the origin board.

```Java
class Solution {
    public boolean res;
    public void dfs(char[][] board,int i,int j, String word,int word_i){
        //越界退出
        if(i<0||j<0||i>=board.length||j>=board[i].length) return;
        //
        if(word.charAt(word_i)!=board[i][j]) return;
        
        else{
            if(word_i==word.length()-1){
                this.res = true;
                return;
            }
            char temp =  board[i][j];
            board[i][j] = '~';
            dfs(board,i+1,j,word,word_i+1);
            dfs(board,i,j+1,word,word_i+1);
            dfs(board,i-1,j,word,word_i+1);
            dfs(board,i,j-1,word,word_i+1);
            //restore the origin board
            board[i][j] = temp;
        }
    }
    public boolean exist(char[][] board, String word) {
        this.res = false;
        for(int i=0;i<board.length;i++){
            for(int j=0;j<board[i].length;j++){
                if(board[i][j]==word.charAt(0)){
                    dfs(board,i,j,word,0);
                } 
                if(this.res==true) return true;
            }
        }
        return false;

    }
}


```
## 678. Valid Parenthesis String
### Abstract: 
### 


```Java

class Solution {
    public boolean checkValidString(String s) {
        if(s=="") return true;
        //low代表未匹配左括号最少的情况
        int low = 0;
        //high代表未匹配左括号最多的情况
        int high = 0;
        for(int i = 0;i<s.length();i++){
            if(s.charAt(i)=='('){
                low++;
                high++;
            } 
            else if(s.charAt(i)==')'){
                low--;
                high--;

            } 
            else if(s.charAt(i)=='*'){
                low--;
                high++;
            } 
            if(high<0) return false;
            if(low<0) low=0;
        }
        
        return low==0;

    }
}

```

## 494. Target Sum
### Abstract: 2D DP, can be converted to Knapsack problem.
### we assume all numbers are postive. And drop certain numbers to reach the difference.
### whenever drop certain numbers, we turn positive to negative, that is 2*num.
### The problem is converted to select some numbers to reach a new target(difference/2).
### we use 1d array, to prevent it from readding, we add from the back.

```Java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum=0;
        for(int num:nums){
            sum += num;
        }
        int diff = sum - target;
        if(diff<0||diff%2!=0) return 0;
        //至此，问题转化成选择填入num满足new_target
        int new_target = diff/2;
        int[] dp = new int[new_target+1];
        dp[0] = 1;
        for(int num:nums){
            //add from the back
            for(int j = new_target;j>=num;j--){
                dp[j] += dp[j-num];
            }
        }


        return dp[new_target];
    }
}

```
## 1049. Last Stone Weight II
### Abstract: 2D DP, can be converted to Knapsack problem.
### Finally, the result = a pile of stones(+) crash another pile(-).
### The problem is to select certain pile(-) to approach sum/2. So that +pile close to sum/s. The last stone weight is the smallest.
### dp[i] = using stones so far, the biggest achieveable weight smaller than i.

```Java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum=0;
        for(int num:stones){
            sum += num;
        }
        int halfSum = sum/2;
        //dp[i]代表不超过i的最大重量
        int[] dp = new int[halfSum+1];
        for(int num:stones){
            for (int i=halfSum;i>=num;i--) {
            dp[i] = Math.max(dp[i],dp[i-num]+num);
            }
            
        }
        
        return sum - 2*dp[halfSum];

    }
}

```

## 322. Coin Change
### Abstract: DP, can be converted to repeatable Knapsack problem.
### dp record the min times to reach i.
```Java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(amount==0) return 0;
        int [] dp = new int[amount+1];
        Arrays.fill(dp,amount+1);
        dp[0] = 0;
        for(int coin:coins){
            for(int i=0;i<amount+1;i++){
                if(i>=coin)dp[i] = Math.min(dp[i],dp[i-coin]+1);
            }
                       
        }
        //System.out.println(dp[amount]);
        return dp[amount]==amount+1?-1: dp[amount];
    }
}
```

## 621. Task Scheduler
### Abstract: Greedy
### 2 sub problems->if many different tasks,then time is tasks.length. If not enough, then we schedule them as AXXAXXA(assume A appears the most,X can be cool down or a task), then time = (max-1)*(n+1)+1? what if B appears the same as A.
### We schedule ABXABXAB, then time = (max-1)*(n+1)+max_count

```Java

class Solution {
    public int leastInterval(char[] tasks, int n) {
        if(n==0) return tasks.length;
        int[] count = new int[26];
        for(char task:tasks){
            count[task-'A']++;
        }
        int max = 0;
        int max_count = 0;
        for(int i = 0;i<count.length;i++){
            if(count[i]>max){
                max = count[i];
                max_count = 1;
            }
            else if(count[i]==max){
                max_count++;
            }
        }
        return Math.max((max-1)*(n+1)+max_count,tasks.length);

    }
}

```

## 138. Copy List with Random Pointer
### Abstract: Deep copy, HashMap 
### First iteration fill HashMap-record original and new node.
### Seconde iteration fill each new node's next, random querying the HashMap.
```Java
class Solution {
    public Node copyRandomList(Node head) {
        if(head==null) return null;
        HashMap<Node,Node> map = new HashMap<>();
        Node cur = head;
        while(cur!=null){
        map.put(cur,new Node(cur.val));
        cur = cur.next;
        }
        cur = head;
        while(cur!=null){
            Node newNode = map.get(cur);
            newNode.random = map.get(cur.random);
            newNode.next = map.get(cur.next);
            cur = cur.next;

        }
        return map.get(head);
        
    }
}
```

## 236. Lowest Common Ancestor of a Binary Tree
### Abstract: consider root node - If root==q or p,then it is LCA. If left and right subtrees both found, then it is LCA. Else if only found in the left subtree, the first found node in the left subtree is LCA.

```Java
class Solution {  
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null) return null;
        if(root==p||root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left!=null&&right!=null) return root;
        if(left!=null) return left;
        if(right!=null) return right;
        else return null;    
    }
}

```
## 207. Course Schedule
### Abstract: topology sort, use HashMap record the mapping relation,1 to many. where to start? we start from the courses without prerequisites, that is inDegree = 0.
### So we record the inDegree of every course. learn basic course, unlock new course when its inDegree decrease to 0.
```Java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int [] inDegree = new int[numCourses];
        HashMap<Integer,List<Integer>> unlock = new HashMap<>();
        //Arrays.fill(graph,-1);
        for(int i=0;i<prerequisites.length;i++){
            int key = prerequisites[i][1];
            int value = prerequisites[i][0];
            inDegree[value]++;
            if(!unlock.containsKey(key)){
                List<Integer> newList = new ArrayList<Integer>();
                newList.add(value);
                unlock.put(key,newList);
            } 
            else unlock.get(key).add(value);
        }
        Queue<Integer> zeroQ = new LinkedList<>();
        for(int i=0;i<inDegree.length;i++){
            if(inDegree[i]==0) zeroQ.offer(i);
        }
        int count = 0;
        while(!zeroQ.isEmpty()){
            int course = zeroQ.poll();
            count++;
            if(!unlock.containsKey(course)) continue;
            for(Integer c:unlock.get(course)){
                if(--inDegree[c]==0) zeroQ.offer(c);
            }  
        }
        return numCourses==count;


    }
}
```
## 332. Reconstruct Itinerary
### Abstract: Map problem as well. HashMap record every departure and its Arrivals. Arrivals should be tried in order, we use priority q.
### Thinking in greedy method, if we have no else place to go, this is the end of the path. For example, we go through a path which has not covered all the tickets, then it should be in the last, if it is a correct path it should be in the last as well.
补充一些正确性的理解：

按照贪心的方式走，如果走到一个点，发现无法继续走了，并且还有某条边没有走过。则说明之前某一个分岔点上走错了，提前进入了一条无法回头的路。应该先走其它边，最后再走这一条无法回头的路。

把这段已知的无法回头的路去掉，则是一个更小的图。我们假想对这个小的图如何走，起点还是和原来一样，终点必然是刚刚的分岔点（因为分岔点不是终点，其原来的度数原来为偶数，去掉一条边后度数为奇数。）。在这个小的图上得到自然排序最小的答案后，再在最后添加上刚刚去掉的那一段，则是大图上的结果。
```Java
class Solution {
    List<String> res = new ArrayList<>();
    Map<String,PriorityQueue<String>> map = new HashMap<>();
    public List<String> findItinerary(List<List<String>> tickets) {
        for (List<String> ticket: tickets) {
            String from = ticket.get(0), to = ticket.get(1);
            PriorityQueue<String> queue = map.get(from);
            if (queue == null) {
                queue = new PriorityQueue<>();
                map.put(from, queue);
            }
            queue.offer(to);
        }
        dfs("JFK");
        return res;
    }

    private void dfs(String from) {
        PriorityQueue<String> tos = map.get(from);
        while (tos != null && tos.size() > 0) {
            dfs(tos.poll());
        }
        res.add(0,from);
    }

}
```
## 684. Redundant Connection
### Abstract: Union-Find set. Firstly, every node is a seperate disjoint set. By reading edges, we merge seperate set.(Each set has a root node, every element treats it as a symbol of their set.) So, when 2 nodes in a edge is in the same set(they have the same symbol), this edge is a redundant connection.
```Java
class Solution {
    int[] set;
    public int find(int u){
        if(u==set[u]) return u;
        set[u] = find(set[u]);
        return set[u];       
    }

    public int[] findRedundantConnection(int[][] edges) {
        int[] res = new int[2];
        set = new int[edges.length+1];
        for(int i = 0;i<set.length;i++){
            set[i] = i;
        }
        for(int i=0;i<edges.length;i++){
            int node0 = find(edges[i][0]);
            int node1 = find(edges[i][1]);
            if(node0==node1) return edges[i];
            else set[node0] = node1;
          
        }
        
        return res;
    }
}
```
## 450. Delete Node in a BST

```Java
class Solution {
    public TreeNode findReplace(TreeNode del,TreeNode cur,boolean right){
        if(right){
            if(cur.right==null){
                del.val = cur.val;
                return cur.left;
            }
            else{
                cur.right = findReplace(del,cur.right,right);
            } 
        }
        else{
            if(cur.left==null){
                del.val = cur.val;
                return cur.right;
            }
            else{
                cur.left = findReplace(del,cur.left,right);
            } 

        }
        return cur;
    }
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return null;
        int val = root.val;
        if(val == key){
            int tmp = root.val;
            if(root.left!=null){
                root.left = findReplace(root,root.left,true);
            }
            else if(root.right!=null){
                root.right = findReplace(root,root.right,false);
            }
            else{
                return null;
            }
        }
        else if(val > key) root.left=deleteNode(root.left,key);
        else root.right=deleteNode(root.right,key);
        return root;

    }
}
```

## 84. Largest Rectangle in Histogram
```Java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int[] temp = new int[heights.length+2];
        System.arraycopy(heights,0,temp,1,heights.length);
        Deque<Integer> stack = new ArrayDeque<>();
        int res = 0;
        for(int i=0;i<temp.length;i++){
            while(!stack.isEmpty()&& temp[stack.peek()]>temp[i]){
                int h = temp[stack.pop()];
                //直到上一个元素，但不包括上一个元素
                int width = i-stack.peek()-1;
                res = Math.max(res,h*width);
            }
            stack.push(i);
        }
        return res;
    }
}
```

## 509. Fibonacci Number
###
```Java
class Solution {
    public int fib(int n) {
        if(n==0) return 0;
        if(n==1) return 1;
        return fib(n-1)+fib(n-2);

    }
}
```
```Java
class Solution {
    public int fib(int n) {
        if(n==0) return 0;
        if(n==1) return 1;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 2;i<=n;i++){
            dp[i] = dp[i-1]+dp[i-2];
        }
        return dp[n];

    }
}
```

## 1035. Uncrossed Lines

```Java
class Solution {
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length+1][nums2.length+1];
        for(int i=1;i<=nums1.length;i++){
            for(int j=1;j<=nums2.length;j++){
                if(nums1[i-1]==nums2[j-1]) dp[i][j] = dp[i-1][j-1]+1;
                else dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);
            }
        }
        return dp[nums1.length][nums2.length];

    }
}
```

## 2379. Minimum Recolors to Get K Consecutive Black Blocks
### Fixed-size sliding window

```Java
class Solution {
    public int minimumRecolors(String blocks, int k) {
        int length = blocks.length();
        int l = 0;
        int r = 0;   
        int count = 0;
        for(;r<k;r++){
            if(blocks.charAt(r)=='W'){
                count++;
            }
        }
        int res = count;
        for(;r<length;l++,r++){
            if(blocks.charAt(r)=='B'&&blocks.charAt(l)=='W'){
                count--;
                res = Math.min(res,count); 
            }
            if(blocks.charAt(r)=='W'&&blocks.charAt(l)=='B'){
                count++;
            }
             

        }
        
        
        
        return res;

    }
}
```
## 763. Partition Labels
```Java
class Solution {
    public List<Integer> partitionLabels(String s) {
        int [] last = new int[26];
        int length = s.length();
        //record last present index
        for(int i=0;i<length;i++){
            last[s.charAt(i)-'a'] = i;
        }
        List<Integer> res = new ArrayList<Integer>();
        //greedy, read and update current substring's end until we can reach the end. This is the shortest substring we can get.
        for(int i=0;i<length;i++){
            int end = last[s.charAt(i)-'a'];
            for(int j = i+1;j<end;j++){
                int newEnd = last[s.charAt(j)-'a'];
                end = Math.max(end,newEnd);
            }
            res.add(end-i+1);
            i = end;
        }
        
        
        
        return res;

    }
}
```
