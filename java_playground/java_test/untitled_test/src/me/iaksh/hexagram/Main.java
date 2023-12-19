package me.iaksh.hexagram;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.Vector;

interface CanSmile{
    void smile();
}

interface CanSpeak {
    void cough();
}

interface CanSayChineseHello extends CanSpeak, CanSmile {
    void sayChineseHello();
}

interface CanSayEnglishHello extends CanSpeak, CanSmile{
    void sayEnglishHello();
}

abstract class AbstractPerson {
    protected final String name;

    public AbstractPerson(String name) {
        this.name = name;
    }

    abstract void cnm();
}

abstract class AbstractSpeakablePerson extends AbstractPerson implements CanSayChineseHello,CanSayEnglishHello {
    public AbstractSpeakablePerson(String name) {
        super(name);
    }

    @Override
    void cnm() {
        System.out.printf("cnm from %s\n",name);
    }
}

//class Person extends AbstractPerson implements CanSayChineseHello,CanSayEnglishHello {
class ParticularPerson extends AbstractSpeakablePerson implements Runnable {
    public ParticularPerson(String name) {
        super(name);
    }

    @Override
    public void sayChineseHello() {
        cough();
        System.out.printf("%s: 你好\n",name);
        smile();
    }

    @Override
    public void sayEnglishHello() {
        cough();
        System.out.printf("%s: hello\n",name);
        smile();
    }

    @Override
    public void smile() {
        System.out.printf("*%s is smiling*\n",name);
    }

    @Override
    public void cough() {
        System.out.printf("%s: *cough*\n",name);
    }

    @Override
    public void run() {
        cough();
        cough();
        sayChineseHello();
        smile();
    }
}

class Base {
    public Base(String name) {
        System.out.printf("Base name = %s\n",name);
    }
}

class Son extends Base {
    public Son() {
        super("idk");
        System.out.println("Son");
    }
}

public class Main {
    public static void main(String[] args) {
        final int[] arr = {1,2,3,4};
        final int[] arr2 = new int[114];
        System.out.println(arr.length);
        System.out.println(100 / 8);
        AbstractSpeakablePerson p = new ParticularPerson("Somebody");
        p.sayChineseHello();
        p.sayEnglishHello();

        Scanner scanner = new Scanner(System.in);
        String input;
        input = scanner.nextLine();
        System.out.println(input
                .replace('?','!')
                .replace('？','!')
                .replace('你','我')
                .replace('吗','哦'));
        p.cnm();

        final Double d = 114.0;
        final Integer i = 2;
        System.out.println(Math.pow(d,i));

        //System.out.println(p instanceof AbstractPerson);
        LinkedList llist = new LinkedList<Integer>();
        Vector v = new Vector<Integer>();
        Thread t = new Thread(new ParticularPerson("the person on a thread"));
        t.setPriority(9);

        /*
        synchronized ("") {
            t.start();
        }
         */

        PrintWriter writer = new PrintWriter(System.out);
        writer.println("caonima");

        new Son();
    }
}