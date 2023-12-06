package me.iaksh.hexagram;

import java.util.Random;

public class RangedRandom {
    private Random random;

    public RangedRandom(Random random) {
        this.random = random;
    }

    public RangedRandom() {
        this.random = new Random();
    }

    public int getBetween(int n,int m) {
        double mean = 24.5;
        double stdDev = 14.5;
        int value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        while (value < n || value > m) {
            value = (int) Math.round(random.nextGaussian() * stdDev + mean);
        }
        return value;
    }
}
