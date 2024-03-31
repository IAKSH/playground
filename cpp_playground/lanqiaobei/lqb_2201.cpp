/**
 * https://www.lanqiao.cn/problems/2201/learning/?page=1&first_category_id=1&second_category_id=3&tags=01%E8%83%8C%E5%8C%85
 * 不会01背包
 * 目前能过11/20, 不能过的全是错误
*/

/*
5
4 4
1 1
5 2
5 5
4 3
=> 10

10
17 26
1 35
13 23
6 97
5 61
15 50
6 16
12 30
7 11
13 86
=> 412
*/

#ifndef OTHERS
#include <bits/stdc++.h>

using namespace std;

bool check(const vector<int>& weights,const vector<int>& prices) {
    int len = weights.size();
    for(int i = 0;i < len;i++) {
        if(accumulate(weights.begin() + i + 1,weights.end(),0) > prices[i]) {
            return false;
        }
    }
    return true;
}

int main() noexcept {
    int n; cin >> n;
    vector<pair<int,int>> v(n);
    for(int i = 0;i < n;i++) {
        cin >> v[i].first >> v[i].second;
    }

    sort(v.begin(),v.end(),[](const pair<int,int>& a,const pair<int,int>& b){
        return a.first + a.second < b.first + b.second;
    });

    vector<int> weights,prices;
    while(!v.empty()) {
        weights.emplace_back(v.back().first);
        prices.emplace_back(v.back().second);
        if(!check(weights,prices)) {
            weights.pop_back();
            prices.pop_back();
        }
        v.pop_back();
    }

    cout << accumulate(prices.begin(),prices.end(),0) << '\n';
    return 0;
}
#else
//本题的主体是01背包的动态规划,但核心是对数据进行排序
//为了让塔尽可能地高,每次应该自上而下地选择价值或重量尽可能小的砖
//若两块砖A和B,weight[A]<weight[B]且value[A]<value[B],则必然把A放在B的上面
//但如果weight[A]<weight[B]且value[A]>value[B](或相反),情况就比较复杂
//假设A放在B上面可行而反之不可行,则有weight[A]<value[B],value[A]<weight[B]
//可得weight[A]<value[B]<value[A]<weight[B],即weight[A]+value[A]<weight[B]+value[B]
//同理可得另一种情况也是如上不等式,故三种情况都可以归并为如上不等式
//故按照weight[A]+value[A]<weight[B]+value[B]实现从小到大排序,再按顺序递推即可 
#include <bits/stdc++.h>

using namespace std;

const int N=1e3+10;
const int M=2e4+10;

struct Brick//砖 
{
    int weight;//重量 
    int value;//价值 
    Brick(int w=0,int v=0):weight(w),value(v){}
};
Brick brick[N];
int dp[M];//dp[i]表示装入的总重量不超过i时所能达到的最大价值 
int n;

bool cmp(const Brick &b1,const Brick &b2)//每块砖按照价值+重量从小到大排序 
{
    return b1.weight+b1.value<b2.weight+b2.value;
}

int main()
{
    int weight_sum=0;
    int ans=0;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)//输入n块砖的价值和重量并计算总重量 
    {
        int w,v;scanf("%d%d",&w,&v);
        brick[i]=Brick(w,v);
        weight_sum+=w;
    }
    
    sort(brick+1,brick+1+n,cmp);//排序优化 
    for(int i=0;i<=n;i++)//依次考虑前0~n块砖 
    {
        for(int j=weight_sum;j>=brick[i].weight;j--)
        //枚举背包当前重量j,j的范围[weight_sum,brick[i].weight](倒序) 
        {
            int w=brick[i].weight,v=brick[i].value;
            //若选择该块砖,则选择之前的背包体积为j-w 
            //若当前砖的价值v不小于上面所有砖的总重量j-w,则可以选择这块砖
            //若选择,则为dp[j-w]+v
            //若不选择,则为dp[j] 
            if(j-w<=v)dp[j]=max(dp[j],dp[j-w]+v);
            ans=max(ans,dp[j]);//记录这一过程中的dp数组最大值 
        }
    }
    printf("%d\n",ans);
    return 0;
}
#endif