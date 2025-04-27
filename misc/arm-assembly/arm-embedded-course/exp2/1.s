        AREA EXAMPLE,CODE,READONLY	;定义本段名称和属性
        ENTRY				;程序入口
        
start
        MOV R0,#10			;置R0为10
        MOV R1,#3			;置R1为3
        ADD R0,R0,R1		        ;将R0置为R0与R1之和
        END				;程序结束