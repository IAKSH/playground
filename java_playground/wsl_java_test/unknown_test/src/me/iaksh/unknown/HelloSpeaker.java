package me.iaksh.unknown;
public class HelloSpeaker {
    private String name;
    public HelloSpeaker(String name) {
        this.name = name;
    }
    public void speak() {
        System.out.printf("%s: 你好\n",name);
    }
}
