// https://leetcode.cn/problems/capitalize-the-title/description/

#include <bits/stdc++.h>

using namespace std;

/**
 * FSM
 *                E
 *              /   \
 * do upper & lower  \ 
 *            /    when title[i] == ' '
 *           /         \
 *          /           \
 *         /       push i forward
 *        /               \
 *       N -- set mv_i --> I
*/
class Solution {
public:
    string capitalizeTitle(string title) {
        for(int i = 0;i < title.size();i++) {
            int mv_i = i;
            for(;i < title.size() && title[i] != ' ';i++);
            int distance = i - mv_i;
            if(distance == 1) {
                title[mv_i] = tolower(title[mv_i]);
            }
            else if(distance == 2) {
                for(;mv_i < i;mv_i++) {
                    title[mv_i] = tolower(title[mv_i]);
                }
            }
            else {
                for(title[mv_i++] = toupper(title[mv_i]);mv_i < i;mv_i++) {
                    title[mv_i] = tolower(title[mv_i]);
                }
            }
        }
        return title;
    }
};

int main() noexcept {
    string s = "IUz g OM";
    Solution solution;
    cout << solution.capitalizeTitle(s) << endl;
    return 0;
}