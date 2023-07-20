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

// 2023-07-15 22:08:16
// 18. 四数之和
// 给定一个长度为n的整数数组和target，找出所有和为target的四元组，假设仅有一个最优解

// 回溯 : 超时，不用了
// 类似于三数之和，先排序，然后固定两个数，然后使用双指针找另外两个数
// 值得注意的是，需要去重，所有指针都需要去重
func fourSum(nums []int, target int) [][]int {
	res := make([][]int, 0)
	sort.Ints(nums)
	for i := 0; i < len(nums)-3; i++ {
		if i > 0 && nums[i] == nums[i-1] { // 去重
			continue
		}
		for j := i + 1; j < len(nums)-2; j++ {
			if j > i+1 && nums[j] == nums[j-1] { // 去重
				continue
			}
			l, r := j+1, len(nums)-1
			for l < r {
				sum := nums[i] + nums[j] + nums[l] + nums[r]
				if sum == target {
					res = append(res, []int{nums[i], nums[j], nums[l], nums[r]})
					for l < r && nums[l] == nums[l+1] { // 去重
						l++
					}
					for l < r && nums[r] == nums[r-1] { // 去重
						r--
					}
					l++
					r--
				} else if sum < target {
					l++
				} else {
					r--
				}
			}
		}
	}
	return res
}

// 2023-07-16 15:30:32
// 834. 树中距离之和
// 给定一个无向、连通的树，树中有n个节点，标记为0到n-1，边一共n-1条
// 给定edges [][]int，表示节点edges[i][0]和edges[i][1]之间有一条边，树的根节点为0，返回一个数组ans，ans[i]表示节点i到其他节点的距离之和

// 先用邻接表存储树，然后使用dfs计算每个节点到其他节点的距离之和

// Graph 邻接表
type Graph struct {
	adj map[int][]int
}

// NewGraph 构造函数
func NewGraph(edges [][]int) *Graph {
	g := &Graph{make(map[int][]int)}
	for _, edge := range edges {
		g.adj[edge[0]] = append(g.adj[edge[0]], edge[1])
		g.adj[edge[1]] = append(g.adj[edge[1]], edge[0]) // 无向图
	}
	return g
}

func sumOfDistancesInTree(n int, edges [][]int) []int {
	size := make([]int, n)
	res := make([]int, n)
	g := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		g[u] = append(g[u], v)
		g[v] = append(g[v], u)
	}
	var dfs func(int, int)
	dfs = func(u, parent int) {
		for _, v := range g[u] {
			if v == parent {
				continue
			}
			dfs(v, u)
			// 递归回来，先计算子节点的距离之和，然后再计算当前节点的距离之和
			size[u] += size[v]
			res[u] += res[v] + size[v]
		}
		size[u]++
	}
	var dfs2 func(int, int)
	dfs2 = func(u, parent int) {
		for _, v := range g[u] {
			if v == parent {
				continue
			}

			res[v] = res[u] - size[v] + n - size[v]
			dfs2(v, u)
		}
	}
	dfs(0, -1)
	dfs2(0, -1)
	return res
}

// 2023-07-17 10:21:18
// 415. 字符串相加
// 给定两个字符串num1和num2，返回两个字符串的和，字符串中只包含数字字符，不包含前导0，不允许使用内置函数，不允许将字符串转换为整数

// 模拟加法，从后往前加，注意进位
func addStrings(num1 string, num2 string) string {
	stack1 := make([]byte, 0)
	stack2 := make([]byte, 0)
	for i := 0; i < len(num1); i++ {
		stack1 = append(stack1, num1[i])
	}
	for i := 0; i < len(num2); i++ {
		stack2 = append(stack2, num2[i])
	}
	res := make([]byte, 0)
	carry := 0
	for len(stack1) > 0 || len(stack2) > 0 {
		a, b := 0, 0
		if len(stack1) > 0 {
			a = int(stack1[len(stack1)-1] - '0')
			stack1 = stack1[:len(stack1)-1]
		}
		if len(stack2) > 0 {
			b = int(stack2[len(stack2)-1] - '0')
			stack2 = stack2[:len(stack2)-1]
		}
		sum := a + b + carry
		res = append(res, byte(sum%10)+'0')
		carry = sum / 10
	}
	if carry > 0 {
		res = append(res, '1')
	}
	for i := 0; i < len(res)/2; i++ {
		res[i], res[len(res)-1-i] = res[len(res)-1-i], res[i]
	}
	return string(res)
}

// 2023-07-19 17:28:06
// 874. 模拟行走机器人
// command 数组给定命令集合 -1 右转 -2 左转 1 <= x <= 9 表示向前移动x个单位长度

// 模拟上下左右行走情况，使用map存储障碍物，使用dx,dy表示上下左右四个方向，使用d表示当前方向，使用x,y表示当前位置
func robotSim(commands []int, obstacles [][]int) int {
	obstacle := make(map[[2]int]bool)
	for _, ob := range obstacles {
		obstacle[[2]int{ob[0], ob[1]}] = true
	}
	dxy := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}} // 上下左右
	d := 0                                           // 当前方向
	x, y := 0, 0                                     // 当前位置
	for _, command := range commands {
		if command == -1 {
			d = (d + 1) % 4
			continue
		}
		if command == -2 {
			d = (d + 3) % 4
			continue
		}
		if command >= 1 && command <= 9 {
			for i := 0; i < command; i++ {
				if _, ok := obstacle[[2]int{x + dxy[d][0], y + dxy[d][1]}]; ok {
					break
				}
				x += dxy[d][0]
				y += dxy[d][1]
			}
		}
	}
	return x*x + y*y
}

// 2023-07-20 10:41:44
// 918. 环形子数组的最大和
// 给定一个由整数数组A表示的环形数组C，求C的非空子数组的最大可能和

// 对于环形数组，可以分为两种情况，一种是不跨越数组首尾，一种是跨越数组首尾
// 不跨越首尾数组的情况直接使用k算法；跨越了首尾的情况，可以转换为求不跨越首尾的情况，即求总和减去中间最小的连续子数组和
func maxSubarraySumCircular(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	curmax, rmax, curmin, rmin := 0, -1<<31, 0, 1<<31
	total := 0
	// 全为负数的情况
	allNegative := true
	for _, n := range nums {
		if allNegative && n > 0 {
			allNegative = false
		}
		curmax = max(curmax+n, n)
		rmax = max(curmax, rmax)
		curmin = min(curmin+n, n)
		rmin = min(rmin, curmin)
		total += n
	}
	if allNegative {
		return rmax
	}
	return max(rmax, total-rmin)
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// code here
//TODO

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
