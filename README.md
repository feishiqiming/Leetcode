# Leetcode
## 704. Binary Search

### Abstract: 
### consider nums = [2], so while condition left <= right instead of left < right, also to avoid dead loop, right = mid -1, left = mid + 1.

tips: while condition left<=right because if left== right then this point remains unsearched.//keep searching when the search window still have numbers.
mid point location = left point location + distance->(right - left)//2    //prevent (left + right)/2 overflow
right = mid - 1; left = mid + 1// ensure search window gets shorter

special condition to be considered: if left == right and nums[mid] != target , if rewrite right = mid left = mid, then search window doesn't get shorter and becomes dead loop.

X^2^
H~2~O
