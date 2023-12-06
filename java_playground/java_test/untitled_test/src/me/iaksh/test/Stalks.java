package me.iaksh.test;

import java.util.Random;

public class Stalks {
    private int[] yao_values;
    private Random random;

    public Stalks()
    {
        random = new Random();
        yao_values = new int[6];
        for(int i = 0;i < 6;i++) {
            yao_values[i] = computeFinal();
        }
    }

    public Hexagram getHostHexagram() {
        int code = 0b000000;
        for(var value : yao_values) {
            code = code << 1 | ((value % 2 != 0) ? 1 : 0);
        }
        return Hexagram.fromCode(code);
    }

    public Hexagram getTransHexagram() {
        int code = 0b000000;
        for(var value : yao_values) {
            if(value == 6) {
                value = 7;
            }
            else if(value == 9) {
                value = 8;
            }
            code = code << 1 | ((value % 2 != 0) ? 1 : 0);
        }
        return Hexagram.fromCode(code);
    }

    private int computeProcess(int stalks) {
        int left = getRandom(0,stalks);
        int leftFinal = left % 4;
        int rightFinal = (stalks - 1 - left) % 4;
        return (leftFinal != 0 ? leftFinal : 4) + (rightFinal != 0 ? rightFinal : 4) + 1;
    }

    private int computeFinal() {
        int stalks = 50;
        for(int i = 0;i < 3;i++) {
            stalks -= computeProcess(--stalks);
        }
        return stalks / 4;
    }

    public int getRandom(int n,int m) {
        double mean = 24.5;
        double stdDev = 14.5;
        int value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        while (value < n || value > m) {
            value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        }
        return value;
    }
}
