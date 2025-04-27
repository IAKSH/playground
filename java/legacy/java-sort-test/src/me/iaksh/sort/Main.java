package me.iaksh.sort;

import java.util.ArrayList;

class Object {
    private final int id;
    private final String name;

    public Object(int id,String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    protected void finalize() {
        System.out.printf("Object(id=%d,name=%s) finalized\n",id,name);
    }
}

public class Main {
    private static <T> void templateTest(T t) {
        System.out.printf("t is %s\n", t.getClass().toString());
    }

    public static void main(String[] args) {
        ArrayList<Object> objects = new ArrayList<Object>();
        for(int i = 0;i < 3;i++) {
            objects.add(new Object(i,String.format("object_%d",i)));
        }
        for(var a : objects.toArray()) {
            System.out.println(a.toString());
        }

        ArrayList<Integer> toSortArray = new ArrayList<Integer>();
        for(int i = 0;i < 114514;i++) {
            toSortArray.add(i);
        }
        toSortArray.sort((x,y) -> {return 0;});
        for(var i : toSortArray) {
            templateTest(i);
        }

        System.out.printf("length of toSortArray = %d\n",toSortArray.size());
    }
}