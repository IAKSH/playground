package me.iaksh.hexagram;

import java.io.PrintWriter;
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
        testSomethingIDK();
        testAutoDePackage();
        testInstanceof();
        testOldCollections();
        testThread();
        whatIsPrintWriter();
        testConstructOrder();
        testDefaultString();
        testStringPlusInt();
        testCharFromHexAndBin();
    }

    private static void testSomethingIDK() {
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
    }

    private static void testAutoDePackage() {
        final Double d = 114.0;
        final Integer _i = 2;
        System.out.println(Math.pow(d,_i));
    }

    private static void testInstanceof() {
        System.out.println(new ParticularPerson("idk") instanceof AbstractPerson);
    }

    private static void testOldCollections() {
        LinkedList llist = new LinkedList<Integer>();
        Vector v = new Vector<Integer>();
        System.out.println(v.size());
        System.out.println(llist.size());
    }

    private static void testThread() {
        Thread t = new Thread(new ParticularPerson("the person on a thread"));
        t.setPriority(9);
        synchronized ("") {
            t.start();
        }
    }

    private static void whatIsPrintWriter() {
        PrintWriter writer = new PrintWriter(System.out);
        writer.println("caonima");
    }

    private static void testConstructOrder() {
        new Son();
    }

    private static void testDefaultString() {
        System.out.println("begin");
        System.out.println(new String());
        System.out.println("end");
    }

    private static void testStringPlusInt() {
        System.out.println("\"1+2=\" + 3 = " + ("1+2=" + 3));
    }

    private static void findRamdomEqualsToZero() {
        while(true) {
            double val = Math.random();
            System.out.println(val);
            if(val == 0) {
                break;
            }
        }
    }

    private static void testCharFromHexAndBin() {
        char hex_c = '\uabcd';
        char bin_c = 0b1010101111001101;
        System.out.println(hex_c);
        System.out.println(bin_c);
    }
}