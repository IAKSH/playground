        AREA EXAMPLE,CODE,READONLY	;定义本段名称和属性
        ENTRY				;程序入口
        
start
        MOV R0,#10			;置R0为10
        MOV R1,#3			;置R1为3
        ADD R0,R0,R1		        ;将R0置为R0与R1之和
        CMP R1,#3                       ;比较R1和3
        ADDHI R0,R0,#1                  ;如果R1大于3，则R0=R0+1
        CMP R1,#11                      ;比较R1和11
        MOVHI R0,#1                     ;如果R1大于11，则R0置为1
        ADDLS R0,R1,#11                 ;如果R1小于11，则R0=R1+11
        END				;程序结束