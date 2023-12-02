#include<stdio.h>
#define Maxsize 20

typedef int ElemType;
typedef struct {
	ElemType data[Maxsize];
	int length;
} seqlist;

int Insert(seqlist *L) {
	int a;
	int i;//位置
	ElemType e;//数字
	printf("输入位置");
	scanf("%d",&i);
	printf("输入数字");
	scanf("%d",&e);
	int k;
	if (L->length==Maxsize) {
		a=0;
	}
	if(i<1||i>L->length+1)
		a=1;
	if (i>=1&&i<L->length) {
		a=2;
		for(k=L->length-1; k>=i-1; k--)
			L->data[k+1]=L->data[k];
	L->data[i-1]=e;
	L->length++;
	}
	if(a==0)
		printf("满");
	else if(a==1)
		printf("不合法") ;
	else if(a==2)
		printf("成功") ;
	MainMenu(L);
}

int Delete(seqlist*L) {
	int a;
	int i;//位置
	ElemType e;//记录删除的数字
	printf("输入位置");
	scanf("%d",&i);
	int k;
	if (L->length==0)
		a=0;
	if (i<1||i>L->length)
		a=1;
	e=L->data[i-1];
	if (i>=1&&i<L->length) {
		a=2;
		for(k=i; k<L->length; k++)
			L->data[k-1]=L->data[k];
	}
	L->length--;
	if(a==0){
		printf("长度为零");
	}else if(a==1)
		printf("位置不合法") ;
	if(a==2){
		printf("success\n",e);
		printf("删除的为%d",e);
	}
	MainMenu(L);
}

void show(seqlist*L) {
	int i;
	for(i=0; i<L->length; i++)
		printf("%d ",L->data[i]);
	printf("\n");
	MainMenu(L);
}
void find(seqlist*L) {
	int i;
	ElemType e;
	printf("输入数字");
	scanf("%d",&e);
	for(i=0; i<L->length; i++) {
		if(L->data[i]==e) {
			printf("%d",i+1);
			break;
		}
	}
	if(i==L->length)
		printf("false");
	MainMenu(L);
}
void MainMenu(seqlist* L) { // 主菜单函数
	int a;
	printf("………………………………………………………\n");
	printf("1.增加数据\n");
	printf("2.显示数据\n");
	printf("3.查找数据\n");
	printf("4.删除数据\n");
	printf("………………………………………………………\n");
	printf("请输入服务前的数字\n");
	scanf("%d",&a);
	switch(a) {
		case 1:
			Insert(L);
			break;
		case 2:
			show(L);
			break;
		case 3:
			find(L);
			break;
		case 4:
			Delete(L);
			break;
	}
	return;
}

int main() {
	seqlist L;
	L.data[0]=1;
	L.data[1]=2;
	L.data[2]=3;
	L.data[3]=4;
	L.length=4;
	MainMenu(&L);
	return 0;
}