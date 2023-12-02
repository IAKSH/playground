#include<stdio.h>

typedef int elemtype;

typedef struct Lnode {
	elemtype data;
	struct Lnode *next;
} listnode;
typedef listnode *linklist;

linklist p;
linklist pre;

void MainMenu(linklist H) { // 脰梅虏脣碌楼潞炉脢媒
	int a;
	printf("隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅\n");
	printf("1.脭枚录脫脢媒戮脻\n");
	printf("2.脧脭脢戮脢媒戮脻\n");
	printf("3.虏茅脮脪脢媒戮脻\n");
	printf("4.脡戮鲁媒脢媒戮脻\n");
	printf("隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅隆颅\n");
	printf("脟毛脢盲脠毛路镁脦帽脟掳碌脛脢媒脳脰\n");
	scanf("%d",&a);
	switch(a) {
		case 1:
			Insert(H);
			break;
		case 2:
			Show(H);
			break;
		case 3:
			Find(H);
			break;
		case 4:
			Delete(H);
			break;
	}
	return;
}

linklist Creat(int num) {
	linklist head;
	head = (linklist)malloc(sizeof(listnode));
	head->next=NULL;
	int n,i;

	printf("脢盲脠毛脢媒戮脻");
	for(i=0; i<num; i++) {
		scanf("%d",&n);
		p=(linklist)malloc(sizeof(listnode));
		p->data=n;
		p->next=head->next;
		head->next=p;
	}
	return head;
}

void Find(linklist head) {
	elemtype x;
	scanf("%d",&x);
	p=head->next;
	while(p!=NULL&&p->data!=x)
		p=p->next;
	if(p==NULL)
		printf("error");
	else if(p->data==x)
		printf("true");
	MainMenu(head);
}

void Show(linklist head) {
	linklist p = head->next;
	while (p != NULL) {
		printf("%d ", p->data);
		p = p->next;
	}
	printf("\n"); 
	MainMenu(head);
}

void Insert(linklist head) {
	int k;
	int i;
	elemtype x;
	printf("脢盲脠毛脦禄脰脙");
	scanf("%d",&i);
	printf("脢盲脠毛脢媒脳脰");
	scanf("%d",&x);
	pre=head;
	k=0;
	while(pre!=NULL&&k<i-1) {
		pre=pre->next;
		k=k+1;
	}
	if(k!=i-1||pre==NULL)
		printf("error");
	else {
		p=(linklist)malloc(sizeof(listnode));
		p->data=x;
		p->next=pre->next;
		pre->next=p;
	}
	MainMenu(head);
}

void Delete(linklist head) {
	int i;
	printf("脢盲脠毛脦禄脰脙");
	scanf("%d",&i);
	pre=head;
	int k=0;
	while(pre->next!=NULL&&k<i-1) {
		pre=pre->next;
		k=k+1;
	}
	if(!(pre->next)&&k<=i-1)
		printf("error");
	else {
		p=pre->next;
		pre->next=p->next;
		free(p);
	}
	MainMenu(head);
}

int main() {
	int i,n,a,num;
	linklist H;
	printf("脢盲脠毛脕麓卤铆鲁陇露脠");
	scanf("%d",&num);
	H=Creat(num);
	MainMenu(H);
}