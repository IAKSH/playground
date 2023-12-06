package me.iaksh.test;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        RangedRandom random = new RangedRandom();
        ArrayList<Casting> castings = new ArrayList<>();
        castings.add(new Stalks(random));
        castings.add(new Coins(random));

        for(Casting casting : castings) {
            casting.exec();
            System.out.println(casting.getHostHexagram().getName());
            System.out.println(casting.getTransHexagram().getName());
        }
    }
}