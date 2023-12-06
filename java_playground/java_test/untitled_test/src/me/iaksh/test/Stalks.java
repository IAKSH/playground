package me.iaksh.test;

import java.util.Random;

public class Stalks extends RandomCasting {
    public Stalks() {
        super();
    }

    public Stalks(RangedRandom random) {
        super(random);
    }

    @Override
    public void exec() {
        for(int i = 0;i < 6;i++) {
            yao_values[i] = computeFinal();
        }
    }

    private int computeProcess(int stalks) {
        int left = random.getBetween(0,stalks);
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
}
