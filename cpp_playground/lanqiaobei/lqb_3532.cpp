// https://www.lanqiao.cn/problems/3532/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=30&tags=2023

#include <bits/stdc++.h>

using namespace std;

#define MINE
//#define OLD_VER
//#define GEN_EXAMPLE

#ifdef GEN_EXAMPLE
#undef MINE
int main() {
    srand(time(0)); // 初始化随机数种子
    int n = 1000; // 测试用例的长度
    cout << n << endl;
    for(int i = 0; i < n; i++) {
        int a = rand() % 10; // 生成一个[0, 9]区间的随机数
        int b = rand() % (2 * 100000) + 1; // 生成一个(0, 2 * 10^5]区间的随机数
        cout << a << " " << b << endl;
    }
    return 0;
}
#endif

#ifdef MINE
int main() noexcept {
#else
int __main() noexcept {
#endif
#ifdef OLD_VER
    array<int,10> nums_cnt{0};
    int n;cin >> n;
    vector<pair<int,int>> v(n);
    for(auto& p : v)
        cin >> p.first >> p.second;
    
    sort(v.begin(),v.end(),[](const pair<int,int>& a,const pair<int,int>& b){
        return a.second < b.second;
    });

    int cost = 0;
    int next_val;
    for(int i = 0;i < n;i++)
        ++nums_cnt[v[i].first];
    // init next_val;
    for(int i = 0;i < n;i++) {
        if(nums_cnt[v[i].first] == 0) {
            next_val = i;
            break;
        }
    }
    int aim = n / 10;
    for(int i = 0;i < n;i++) {
        if(nums_cnt[v[i].first] > aim) {
            --nums_cnt[v[i].first];
            ++nums_cnt[next_val];
            v[i].first = next_val;
            cost += v[i].second;
            // update next_val;
            for(int j = 0;j < n;j++) {
                if(nums_cnt[v[j].first] == 0) {
                    next_val = j;
                    break;
                }
            }
        }
    }
    cout << cost << '\n';
    return 0;
#else
    array<int,10> cnts{0};
    int n;cin >> n;
    vector<pair<int,int>> v(n);
    for(auto& p : v) {
        cin >> p.first >> p.second;
        ++cnts[p.first];
    }
    
    sort(v.begin(),v.end(),[](const pair<int,int>& a,const pair<int,int>& b){
        return a.second < b.second;
    });

    int cost = 0;
    int aim = n / 10;
    for(const auto& i : v) {
        if(cnts[i.first] > aim) {
            --cnts[i.first];
            cost += i.second;
        }
    }
    cout << cost << '\n';
    return 0;
#endif
}

#include<bits/stdc++.h>

#define rep(i, a, b) for(int i = a; i < b; i++)
#define x first 
#define y second

using namespace std;

typedef pair<int, int> PII;
typedef long long LL;
const int N = 1e5 + 10;

PII a[N]; //cost num
int dic[20]; //每个数字实际出现的次数
int n;

#ifndef MINE
#ifndef GEN_EXAMPLE
int main()
#else
int ___main()
#endif
#else
int ___main()
#endif
{
    cin >> n;
    rep(i, 1, n + 1)
        cin >> a[i].y >> a[i].x, dic[a[i].y] ++ ; 
    
    sort(a + 1, a + 1 + n); //按照cost从小到大进行排序
    
    LL res = 0;       
    int cnt = n / 10;  //每个数字应该出现的次数
    rep(i, 1, n + 1)
    {
        if(dic[a[i].y] > cnt)
        {
            res += a[i].x;
            dic[a[i].y] --;   
        }
    }
    
    cout << res << endl;
    cout << "this is others\n";
    return 0;
}

/*
some examples:

10
1 1
1 2
1 3
2 4
2 5
2 6
3 7
3 8
3 9
4 10
=27

10
1 1
4 2
1 3
2 4
2 5
4 6
4 7
2 8
3 9
4 10
=25

10
1 1
1 2
4 3
5 4
1 5
4 6
1 7
9 8
1 9
9 10
=26

1000
1 9976
2 8343
6 22435
4 5096
7 598
5 13537
9 22043
9 22255
2 20199
2 19423
5 29717
4 13812
7 23315
9 16439
9 16988
4 12559
0 22694
6 2833
6 1116
3 12313
4 16206
5 15002
8 23208
3 306
5 16160
1 30850
8 32116
8 26428
3 17172
3 8317
1 1819
7 14430
8 900
1 26310
2 20123
9 3202
2 10417
8 3735
6 21012
1 31702
8 32063
5 12466
7 12621
1 27983
6 12468
9 4571
2 14383
0 29985
3 10748
0 2395
9 31048
4 17224
6 27112
6 30591
5 30793
1 8707
8 32170
2 19479
1 25736
2 1108
0 346
7 7416
3 25471
9 9345
3 12917
3 19579
8 27467
7 12950
3 4697
1 16379
2 11566
9 4832
1 6346
4 9818
4 4110
1 11379
3 30320
5 20093
1 15091
1 18342
8 26455
4 3532
9 12172
2 31890
0 5485
5 14003
6 29867
3 12694
8 2823
9 7971
2 20677
4 10710
1 13661
7 14097
7 5235
6 9143
3 30473
6 12022
0 29940
7 1892
7 19084
1 11854
7 30376
2 9548
3 29898
8 6681
6 26518
7 1186
3 18313
8 27485
0 27111
6 16860
7 29685
1 5538
9 3689
7 32053
2 27935
3 32117
7 32525
9 22827
5 23161
0 32485
7 14697
5 2306
2 6748
1 1033
2 22839
4 28479
3 21634
4 11128
9 29396
9 19501
0 7308
3 32463
7 32194
8 31835
7 9342
1 14639
1 20263
7 18204
9 32574
0 7612
9 9292
8 15621
9 29464
9 24095
0 15646
2 9843
2 23613
6 4357
4 1712
3 515
9 7999
7 14494
5 21942
1 4888
0 11159
5 22814
0 12556
6 89
1 3615
6 8000
0 11309
1 560
2 5016
5 5822
8 20925
8 8524
8 31526
4 28890
0 7680
3 1311
5 14543
5 23256
6 2400
5 21089
0 18431
6 25730
8 8939
5 6539
3 3591
8 30371
0 15058
6 21853
9 19609
1 22924
6 7404
1 31065
7 9294
6 19346
2 23344
9 4165
3 10017
9 10126
9 10303
5 27309
7 31473
5 7003
3 16856
4 23903
6 3074
2 28252
6 27489
6 4077
1 20348
8 22515
2 30821
2 23342
5 5611
1 26123
5 26384
4 14269
3 28094
5 29583
2 2741
2 15481
0 5576
9 12017
3 17993
1 11523
5 32706
6 12452
0 22245
9 6212
6 25286
7 8610
4 11848
3 1764
6 24021
0 29159
7 23891
0 14804
0 10835
5 7924
1 3723
5 30917
8 24198
0 2637
6 4250
6 28846
8 16411
4 24571
9 28878
4 8778
2 9888
0 3338
6 9624
2 24727
8 28171
3 17932
4 18703
3 24409
5 31025
6 7573
4 25451
8 20885
9 17488
7 10941
3 17802
3 6166
4 1926
5 16289
6 29550
3 32747
4 26968
9 12258
1 7869
8 4327
3 16684
7 10020
6 12607
1 27680
3 27087
2 3984
1 26468
2 14961
1 9199
8 2131
4 31420
9 11000
7 2210
3 27677
1 23677
3 408
8 3038
7 26531
9 32768
1 13688
4 3838
5 8222
4 13636
8 8066
0 835
0 27081
9 4970
8 14565
8 7249
2 3686
9 29495
1 27027
4 4782
7 9146
0 31125
5 26308
6 30279
6 29196
2 12818
4 8562
9 4331
1 27307
5 14687
5 17627
6 22733
0 11877
8 30675
7 17665
6 29446
9 9232
4 27622
1 32045
3 17937
9 7941
1 1206
2 24168
0 7245
7 5536
7 3205
5 11777
3 22142
1 26561
0 9017
9 21514
1 11853
9 13646
9 18651
2 14811
7 2243
3 17583
8 30744
8 14477
6 22738
8 31056
2 20675
2 6016
5 4557
1 23075
1 21073
5 6869
0 10730
0 10267
9 9820
2 8693
9 11463
4 1202
4 31521
8 22024
4 24034
9 10340
2 28161
6 26751
4 27424
7 10546
2 1540
6 24971
3 12103
4 20768
9 15953
3 32419
4 1156
4 1207
2 8375
1 8450
2 9958
1 14724
9 28711
1 19495
4 9952
0 31895
4 25400
4 13395
1 27595
7 2309
8 28322
6 23823
1 18018
7 28262
8 8131
1 18339
5 4932
9 18263
6 1822
1 12703
9 28022
3 32761
3 13014
3 25430
6 28894
4 11377
5 14647
7 3678
0 3273
4 28518
0 21294
8 15378
8 26214
4 28806
4 1740
6 19472
9 19272
5 25345
6 25031
0 700
7 13288
4 7180
3 17400
8 10171
5 22049
3 14688
2 14814
7 13432
5 18354
5 22456
3 26394
6 4770
6 12480
7 22871
4 241
7 27308
3 3665
4 15487
8 22734
6 588
1 1234
6 507
4 30397
7 11004
1 17042
3 9110
4 17392
9 16508
2 29253
6 6818
1 17726
6 14158
4 15746
7 30322
2 26015
1 15161
7 24388
0 736
0 20658
6 24007
9 18029
5 32456
6 23863
5 21703
7 18494
9 25226
0 1941
0 18635
9 25313
6 13964
4 31829
3 13227
8 4064
6 31158
5 12964
4 20224
5 9899
2 6133
7 5845
4 4778
1 20245
3 28857
6 17629
6 17227
2 983
6 28022
1 21206
1 4298
6 9484
4 10278
4 11133
7 17616
6 6835
4 8645
8 30900
5 334
4 1841
8 2260
5 7680
2 3829
6 2509
2 4056
9 17164
5 15293
9 4401
6 21464
0 28196
7 18783
6 15855
3 21218
8 11962
9 27611
3 21754
8 29318
4 22501
8 22514
0 16937
7 5720
9 19165
6 25005
9 10936
9 18035
7 20865
0 10964
2 4038
8 26097
9 2257
8 24952
9 216
6 31092
9 23412
6 3690
8 24148
9 32161
2 3123
6 29633
6 11271
1 19872
0 20604
0 31576
7 29600
5 6557
4 31456
2 13645
9 14131
9 27977
5 13717
1 26244
1 29623
6 18009
8 19701
5 10018
8 13192
6 15110
7 2557
1 4370
5 12517
0 12427
6 7375
0 32387
9 2163
6 8619
9 5936
9 22632
8 31673
8 29993
1 30475
6 29943
5 17314
1 25678
5 16620
1 1579
7 29438
8 29030
3 10886
8 6210
9 26880
7 28227
1 21805
4 8255
7 21441
2 4792
6 6236
9 6761
7 17084
8 8991
3 23601
4 14538
9 18769
1 9940
8 1004
6 8625
2 12611
2 11960
8 26740
3 15921
3 15673
1 11183
6 25298
5 6304
8 31552
7 21794
3 26292
6 19224
4 10055
1 8061
1 11499
8 32249
5 7586
1 17074
0 28142
4 30309
7 23944
5 20661
2 11348
2 13507
5 7061
1 23550
2 3934
5 9021
9 25673
9 11787
1 19818
3 21569
6 20102
4 3547
6 9857
6 20740
1 15894
2 5747
9 28363
6 30714
1 13859
6 7516
4 10517
6 27488
1 15293
4 10531
3 12626
6 4018
6 18520
3 30680
9 30486
7 6709
5 19531
8 5520
1 18461
6 12143
7 16722
4 740
8 10946
4 4521
6 19724
2 27503
5 18569
1 24436
0 29566
0 27732
0 26774
5 25121
2 22927
8 27317
4 31283
0 27719
7 6026
8 31293
9 12110
0 18497
6 7071
5 21399
4 9976
0 13998
4 7266
2 23285
3 10215
3 31983
7 6986
8 13962
7 27381
8 941
3 30515
1 29968
1 30111
0 1360
3 3805
9 17475
4 1159
9 19593
9 29546
4 14179
0 1523
3 712
5 10980
6 7606
1 9291
7 3783
4 693
1 26884
7 24735
0 765
9 29229
6 9242
2 15647
9 7212
8 3328
8 31047
5 30624
8 20436
1 1477
9 18658
2 21534
7 27325
3 16504
8 20418
6 30516
1 16241
0 29945
3 24004
1 20646
7 25791
5 24971
6 7241
4 5592
4 1082
5 3413
5 25781
3 19540
0 14520
1 6885
2 9969
8 25672
1 31956
1 10920
1 24554
5 25350
7 10604
1 19827
1 8208
5 7554
3 1059
3 1789
9 8664
7 727
1 24032
4 24674
4 28730
4 30402
8 16717
1 2722
4 7476
9 15397
5 12633
5 4383
5 2117
1 28550
6 30646
6 7203
3 13554
4 4493
6 860
6 17240
4 12613
1 8586
3 22276
3 28750
8 3145
6 18413
5 1966
1 20729
2 23695
5 4300
3 19228
7 23089
1 22188
3 13920
0 5717
3 23088
4 29625
7 31624
2 19490
9 6934
6 10194
8 5700
4 31636
7 5480
5 10391
8 10811
4 11900
2 12941
6 17326
7 21303
9 18418
1 8016
1 17727
4 16292
6 29391
1 11873
0 28278
6 23316
8 8243
0 19156
8 31749
6 2237
6 5631
2 31393
3 15519
2 28153
1 3420
2 28962
3 14206
5 19654
7 24882
4 12696
0 25310
5 28228
3 19634
7 16805
7 27264
5 7481
5 14797
8 27603
6 1576
9 12432
9 8122
4 4620
7 24607
9 9686
3 29044
9 4129
7 2993
3 19388
4 7225
9 11844
5 2992
8 31896
5 5645
2 9101
2 26864
5 26851
3 31797
7 10360
9 22729
5 29116
5 16828
3 6501
9 29592
9 4964
1 28146
4 10519
8 31550
8 24512
1 27368
5 9014
0 12704
8 2266
8 25589
3 16077
7 16165
6 31584
5 13363
3 204
1 10555
7 2340
1 25859
7 11911
7 22738
2 9114
8 15493
6 20533
1 12511
4 9821
4 23585
5 17267
8 13530
4 12714
4 28417
5 23999
3 16193
6 5348
9 26772
9 7933
4 18451
8 18702
8 21261
7 29684
7 15190
4 31572
7 11745
0 2258
2 24465
3 10115
4 23629
7 11172
1 19539
1 17036
4 9166
6 16095
9 28842
0 29957
3 15015
6 15322
5 10668
9 26023
6 26207
1 17746
4 26866
8 5073
7 10815
4 18916
4 27416
4 16900
6 1496
7 13974
3 32318
2 28773
3 17276
0 14671
1 22666
8 26431
7 22208
2 4955
0 32420
0 20862
4 19201
9 31720
4 14905
4 26420
1 16859
6 4747
8 8312
0 10264
4 435
0 25019
3 10392
5 9730
2 5167
5 29013
0 21072
7 24432
9 29880
0 26903
7 23645
3 10756
1 20223
6 17189
3 2816
8 11524
2 19829
5 3719
4 32381
6 27609
8 30674
9 16808
9 6557
6 24721
2 8591
8 7384
4 30765
4 9821
3 26731
6 7371
4 30077
8 13331
7 17130
7 32025
8 17122
1 9900
5 7889
1 19132
5 3636
6 7111
1 4012
7 1685
3 14416
8 7205
8 3400
6 22503
1 7064
4 5676
7 16953
7 20186
5 25198
9 26724
1 27221
2 14073
8 28092
1 31740
5 16695
6 31157
6 19311
5 28486
3 189
8 4320
9 15608
0 20534
2 28555
0 2354
7 19914
0 26343
9 32646
8 5320
9 31198
1 4119
9 31833
8 11680
0 25650
8 10150
0 18931
=118146
*/