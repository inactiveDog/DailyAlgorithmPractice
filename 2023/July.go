package _023

import "sort"

// 2023-07-11 21:13:43
// 1911. 最大子序列交替和
// 下标从0开始的数组的 交替和 定义为 偶数 下标处元素之 和 减去 奇数 下标处元素之 和
// 比方说，数组 [4,2,5,3] 的交替和为 (4 + 5) - (2 + 3) = 4
// 输入数组 nums ，返回 nums 中 最大子序列交替和

// 动态规划
// 子序列交替和可以理解为偶数下标为正，奇数下标为负，求最大子序列和
// dp[i][0] 表示以 nums[i] 结尾的交替和的最大值，且 nums[i] 必须选取
// dp[i][1] 表示以 nums[i] 结尾的交替和的最大值，且 nums[i] 必须不选取
func maxAlternatingSum(nums []int) int64 {
	n := len(nums)
	dp := make([][2]int64, n)
	dp[0][0] = int64(nums[0])
	dp[0][1] = 0
	// dp[i][0] = max(dp[i-1][0], dp[i-1][1]+nums[i])
	for i := 1; i < n; i++ {
		dp[i][0] = max64(dp[i-1][0], dp[i-1][1]+int64(nums[i]))
		dp[i][1] = max64(dp[i-1][1], dp[i-1][0]-int64(nums[i]))
	}
	return max64(dp[n-1][0], dp[n-1][1])
}

// 2023-07-12 19:41:06
// 2544.数字的交替和
// 给定一个正整数 n ，最高有效位赋+，其余位数依次赋-，+

// 用一下闭包
func alternateDigitSum(n int) int {
	nums := make([]int, 0)
	for n > 0 {
		nums = append(nums, n%10)
		n /= 10
	}
	res := 0
	changer := func() func() int {
		x := -1
		return func() int {
			return -x
		}
	}
	f := changer()
	for i := len(nums) - 1; i >= 0; i-- {
		res += f() * nums[i]
	}
	return res
}

// 2023-07-13 21:41:19
// 931. 下降路径最小和
// 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和
// 下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素，直到到达最后一行为止，每次下降可以选择正下方或者正下方相邻的元素

// 回溯 : 依次遍历每一行的每一个元素，然后向下遍历
// dp : 创建一个dp数组，dp[i][j] 表示到达 matrix[i][j] 的最小路径和
func minFallingPathSum(matrix [][]int) int {
	n := len(matrix)
	res := 1 << 31
	/** 回溯 : 超时
	// 遍历需要传递所在的行数和当前的和
	var dfs func(int, int, int)
	dfs = func(row, col, sum int) {
		if row == n {
			res = min(res, sum)
			return
		}
		sum += matrix[row][col]
		for i := -1; i <= 1; i++ {
			if col+i >= 0 && col+i < n { // 判断是否越界
				dfs(row+1, col+i, sum)
			}
		}
	}
	for i := 0; i < n; i++ {
		dfs(0, i, 0)
	}
	*/
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	// init
	for i := 1; i < n; i++ {
		for j := 0; j < n; j++ {
			dp[i][j] = res
		}
	}
	for i := 0; i < n; i++ {
		dp[0][i] = matrix[0][i]
	}
	// dp
	for i := 1; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := -1; k <= 1; k++ {
				if j+k >= 0 && j+k < n {
					dp[i][j] = min(dp[i][j], dp[i-1][j+k]+matrix[i][j])
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		res = min(res, dp[n-1][i])
	}
	return res
}

// 2023-07-14 14:11:10
// 979. 在二叉树中分配硬币
// 给定一个有 n 个节点的二叉树的根节点 root ，树中的每个节点上都对应有 node.val 枚硬币，并且总共有 n 枚硬币
// 返回所有节点中的硬币均为1枚的最小操作次数

// 递归 : 从叶子节点开始，如果节点的硬币数为0，则需要从父节点移动硬币，如果节点的硬币数大于1，则需要移动硬币到父节点
// 当前节点需要的移动次数为 node.val - 1 的绝对值
func distributeCoins(root *TreeNode) int {
	res := 0
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		res += abs(left) + abs(right)
		return node.Val + left + right - 1
	}
	dfs(root)
	return res
}

//  16. 最接近的三数之和
// 给定一个长度n的整数数组和target，找出三个数之和最接近target，假设仅有一个最优解

// 回溯 : 边界条件为 len(path) == 3
// 递归 : 从当前位置开始，遍历数组，将当前元素加入path，然后递归，最后将当前元素从path中移除
// 又超时了
// 双指针 : 先排序，然后遍历数组，将当前元素作为第一个元素，然后使用双指针找到剩下的两个元素
func threeSumClosest(nums []int, target int) int {
	sort.Ints(nums)

	closest := nums[0] + nums[1] + nums[2]
	/**path := make([]int, 0)
	closest := 1 << 31
	var dfs func(int, int)
	dfs = func(idx, sum int) {
		if len(path) == 3 {
			if abs(sum-target) < abs(closest-target) {
				closest = sum
			}
			return
		}
		for i := idx; i < len(nums); i++ {
			path = append(path, nums[i])
			dfs(i+1, sum+nums[i])
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0)*/
	// 固定一个，双指针找另外两个

	for i := 0; i < len(nums)-2; i++ {
		l, r := i+1, len(nums)-1
		for l < r {
			sum := nums[i] + nums[l] + nums[r]
			closest = getClosest(closest, sum, target)
			if sum == target {
				return target
			} else if sum < target {
				l++
			} else {
				r--
			}
		}
	}
	return closest
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	return -max(-a, -b)
}
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func getClosest(a, b, target int) int {
	if abs(a-target) < abs(b-target) {
		return a
	}
	return b
}
