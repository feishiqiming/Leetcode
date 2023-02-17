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
