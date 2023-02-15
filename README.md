# Leetcode
## 704. Binary Search

### Abstract: 
### consider nums = [2], so while condition left <= right instead of left < right, also to avoid dead loop, right = mid -1, left = mid + 1.

tips: while condition left<=right because if left== right then this point remains unsearched.//keep searching when the search window still have numbers.
mid point location = left point location + distance->(right - left)//2    //prevent (left + right)/2 overflow
right = mid - 1; left = mid + 1// ensure search window gets shorter

special condition to be considered: if left == right and nums[mid] != target , if rewrite right = mid left = mid, then search window doesn't get shorter and becomes dead loop.

## 704. Binary Search
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
