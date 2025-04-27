package me.iaksh;

import org.lwjgl.openal.*;

import java.util.concurrent.Callable;
import java.util.function.Function;

public class Mixer {
    private ALCCapabilities alcCapabilities;
    private ALCapabilities alCapabilities;
    private long device;
    private long context;

    private Cluster cluster;
    private Channel[] channels;
    private int bpm;

    private void initOpenAL() {
        String defaultDeviceName = ALC11.alcGetString(0, ALC11.ALC_DEFAULT_DEVICE_SPECIFIER);
        device = ALC11.alcOpenDevice(defaultDeviceName);

        int[] attributes = {0};
        context = ALC11.alcCreateContext(device, attributes);
        ALC11.alcMakeContextCurrent(context);

        alcCapabilities = ALC.createCapabilities(device);
        alCapabilities = AL.createCapabilities(alcCapabilities);

        if(!(alCapabilities.OpenAL10 && alCapabilities.OpenAL11)) {
            System.out.println("OpenAL 10/11 is not supported in current context.");
        }
    }

    private void initClaster() {
        cluster = new Cluster(44100);
    }

    private void initChannels() {
        channels = new Channel[4];
        for(int i = 0;i < channels.length;i++) {
            channels[i] = new Channel();
        }
    }

    private void destroyChannels() {
        for(Channel channel : channels)
            channel.destroyAlSource();
    }

    private void destroyOpenAL() {
        ALC11.alcDestroyContext(context);
        ALC11.alcCloseDevice(device);
    }

    private int convertFreq(Note note) {
        double[] equalTemperaments = {0,261.63,293.66,329.63,349.23,392.00,440.00,493.88};
        double baseFreq = equalTemperaments[note.getSimpleScore()];
        double freq = baseFreq *
                Math.pow(2, note.getOctaveShift()) *
                Math.pow(2, note.getSemitoneShift() / 12.0);
        return (int) freq;
    }

    private int convertDurationMs(Sheet sheet) {
        Section section = sheet.currentSection();
        return (int) (section.currentNote().getNoteFraction() * section.getStandardNote() * 60000.0f / bpm);
    }

    private void prepareChannelSquare(Channel channel,Sheet sheet) {
        Section section = sheet.currentSection();
        Note note = section.currentNote();
        channel.setGain(note.getGain());
        channel.bindBuffer(cluster.genSquare(convertFreq(note)));
        channel.setDurationMs(convertDurationMs(sheet));
    }

    private void prepareChannelTriangle(Channel channel,Sheet sheet) {
        Section section = sheet.currentSection();
        Note note = section.currentNote();
        channel.setGain(note.getGain());
        channel.bindBuffer(cluster.genTriangle(convertFreq(note)));
        channel.setDurationMs(convertDurationMs(sheet));
    }

    private void prepareChannelNoise(Channel channel,Sheet sheet) {
        Section section = sheet.currentSection();
        Note note = section.currentNote();
        channel.setGain(note.getGain());
        channel.bindBuffer(cluster.genWhiteNoise(convertFreq(note)));
        channel.setDurationMs(convertDurationMs(sheet));
    }

    private void playAllChannel() {
        long startTimestamp = System.currentTimeMillis();
        for(Channel channel : channels)
                channel.play();

        while(System.currentTimeMillis() - startTimestamp < (long)(60000.0f / bpm)) {
            for(Channel channel : channels)
                    channel.tryStop();
        }

        for(Channel channel : channels)
                channel.stop();
    }

    private boolean allSheetTerminated(Sheet[] sheets) {
        for(Sheet sheet : sheets) {
            if(sheet.eof())
                return true;
        }
        return false;
    }

    private boolean currentSectionsTerminated(Sheet[] sheets) {
        for(Sheet sheet : sheets) {
            if(sheet.currentSection().eof())
                return true;
        }
        return false;
    }

    public Mixer(int bpm) {
        if(bpm < 0)
            throw new IllegalArgumentException();
        this.bpm = bpm;

        initOpenAL();
        initClaster();
        initChannels();
    }

    public void play(Sheet[] sheets) {
        if(sheets.length != 4)
            throw new IllegalArgumentException(String.format("need 4 sheets but given %d",sheets.length));
        while(!allSheetTerminated(sheets)) {
            while(!currentSectionsTerminated(sheets)) {
                prepareChannelSquare(channels[0],sheets[0]);
                prepareChannelSquare(channels[1],sheets[1]);
                prepareChannelTriangle(channels[2],sheets[2]);
                prepareChannelNoise(channels[3],sheets[3]);
                for(Sheet sheet : sheets)
                    if(!sheet.currentSection().eof())
                        sheet.currentSection().nextNote();
                playAllChannel();
            }
            for(Sheet sheet : sheets)
                sheet.nextSection();
        }
    }

    public void destroy() {
        destroyChannels();
        destroyOpenAL();
    }
}
