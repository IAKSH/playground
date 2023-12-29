package me.iaksh;

public class FreqSample {
    private int freq;
    private Integer ms;

    public FreqSample(short freq, int ms) {
        if(ms < 0) {
            throw new RuntimeException(String.format("Delay must be positive num, but given %d",ms));
        }
        this.freq = freq;
        this.ms = ms;
    }

    public int getFreq() {
        return freq;
    }

    public int getMs() {
        return ms;
    }
}
