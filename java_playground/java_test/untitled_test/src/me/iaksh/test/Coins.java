package me.iaksh.test;

public class Coins extends RandomCasting {
    public Coins() {
        super();
    }

    public Coins(RangedRandom random) {
        super(random);
    }

    @Override
    public void exec() {
        for(int i = 0;i < 6;i++) {
            yao_values[i] = computeFinal();
        }
    }

    private int computeFinal() {
        int res = 0;
        for(int i = 0;i < 3;i++) {
            res += random.getBetween(2,3);
        }
        return res;
    }
}
