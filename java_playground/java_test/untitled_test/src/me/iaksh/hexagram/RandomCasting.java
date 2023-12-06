package me.iaksh.hexagram;

public abstract class RandomCasting implements Casting {
    protected RangedRandom random;

    public RandomCasting() {
        random = new RangedRandom();
    }

    public RandomCasting(RangedRandom random) {
        this.random = random;
    }

    @Override
    abstract public void exec();
}
