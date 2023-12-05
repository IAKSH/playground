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

    private int computeProcess() {
        int left = getRandom();
        int left_final = left % 4;
        int right_final = (48 - left) % 4;
        int result = (left_final != 0 ? left_final : 4) + (right_final != 0 ? right_final : 4) + 1;
        return result;
    }

    private int computeFinal() {
        int res = 49;
        for(int i = 0;i < 3;i++) {
            int j = computeProcess();
            System.out.println(j);
            res -= j;
        }
        return res / 4;
    }

    public int getRandom() {
        double mean = 24.5;
        double stdDev = 14.5;
        int value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        while (value < 0 || value > 49) {
            value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        }
        return value;
    }
}
