void heap_sort(int* arr,int length) {
        // sort arr with heap sort
        int i,j,temp;
        for(i=length/2;i>=0;i--) {
            heapify(arr,length,i);
        }
        for(i=length-1;i>=0;i--) {
            temp=arr[0];
            arr[0]=arr[i];
            arr[i]=temp;
            heapify(arr,i,0);
        }
    }
    void heapify(int* arr,int length,int i) {
        int left=2*i+1,right=2*i+2,largest;
        if(left<length&&arr[left]>arr[i]) {
            largest=left;
        } else {
            largest=i;
        }
        if(right<length&&arr[right]>arr[largest]) {
            largest=right;
        }
        if(largest!=i) {
            int temp=arr[i];// phi-2's original code is: "temp=arr[i];"
            arr[i]=arr[largest];
            arr[largest]=temp;
            heapify(arr,length,largest);
        }
    }
    void print_heap(int* arr,int length) {
        int i;
        for(i=0;i<length;i++) {
            printf("%d ",arr[i]);
        }
        printf("\n");
    }
    int main() {
        int arr[]={5,3,1,4,2};
        int length=sizeof(arr)/sizeof(arr[0]);
        print_heap(arr,length);
        heap_sort(arr,length);
        print_heap(arr,length);
        return 0;
    }
