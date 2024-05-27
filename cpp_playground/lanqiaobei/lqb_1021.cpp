// https://www.lanqiao.cn/problems/1021/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92&sort=pass_rate&asc=0
// 一维DP

#include <bits/stdc++.h>

using namespace std;

const string input = //"lanqiao";
"tocyjkdzcieoiodfpbgcncsrjbhmugdnojjddhllnofawllbhf"\
"iadgdcdjstemphmnjihecoapdjjrprrqnhgccevdarufmliqij"\
"gihhfgdcmxvicfauachlifhafpdccfseflcdgjncadfclvfmad"\
"vrnaaahahndsikzssoywakgnfjjaihtniptwoulxbaeqkqhfwl";

int main() {
    ios::sync_with_stdio(false);

    array<int,200> dp;
    fill(dp.begin(),dp.end(),1);
    // dp[i]是以input[i]结尾的最长本质上升序列的计数
    // 因为单个字符也是本质上升，所以dp初始全为1

    int i,j,len = input.size();
    for(i = 0;i < len;i++) {
        for(j = 0;j < i;j++) {
            if(input[j] < input[i])
                dp[i] += dp[j];
            else if(input[j] == input[i])
                dp[i] -= dp[j];
        }
    }

    cout << accumulate(dp.begin(),dp.begin() + len,0) << '\n';
    return 0;
}