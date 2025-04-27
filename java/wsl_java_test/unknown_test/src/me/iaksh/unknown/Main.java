package me.iaksh.unknown;
import me.iaksh.unknown.HelloSpeaker;

import java.util.Vector;

/*
我到底是有什么大病才会想在Win上用IDEA在WSL1上写Java
卡爆了，IDEA每次构建之前都要准备什么WSL构建环境
这个过程死慢
而且还会有一些奇奇怪怪的bug，比如不会自动重新编译
 */

public class Main {
    public static void main(String[] args) {
        Vector<HelloSpeaker> speakers = new Vector<HelloSpeaker>();
        for(int i = 0;i < 10;i++) {
            speakers.add(new HelloSpeaker(String.format("speaker%d",i)));
        }
        for(var speaker : speakers) {
            speaker.speak();
        }
    }
}