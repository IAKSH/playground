package me.iaksh;

import org.lwjgl.openal.*;

public class Mixer {
    private ALCCapabilities alcCapabilities;
    private ALCapabilities alCapabilities;
    private long device;
    private long context;

    private Cluster claster;
    private Channel[] channels;
    private Sheet sheet;

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
        claster = new Cluster(44100);
    }

    private void initChannels() {
        channels = new Channel[4];
        for(int i = 0;i < channels.length;i++) {
            channels[i] = new Channel();
        }
    }

    private void initSheet() {
        sheet = new Sheet();
    }

    private void destroyChannels() {
        for(Channel channel : channels)
            channel.destroyAlSource();
    }

    private void destroyOpenAL() {
        ALC11.alcDestroyContext(context);
        ALC11.alcCloseDevice(device);
    }

    public Mixer() {
        initOpenAL();
        initClaster();
        initChannels();
        initSheet();
    }

    public void play() {
        try {
            sheet.rewind();
            while(!sheet.eof()) {
                Frame frame = sheet.nextFrame();
                channels[0].setGain(frame.getSq0().getGain());
                channels[0].bindBuffer(claster.genSquare(frame.getSq0().getFrequency()));
                channels[1].setGain(frame.getSq1().getGain());
                channels[1].bindBuffer(claster.genSquare(frame.getSq1().getFrequency()));
                channels[2].setGain(frame.getTri().getGain());
                channels[2].bindBuffer(claster.genTriangle(frame.getTri().getFrequency()));
                channels[3].setGain(frame.getNoise().getGain());
                channels[3].bindBuffer(claster.genWhiteNoise(frame.getNoise().getFrequency()));

                for(Channel channel : channels)
                    channel.play();

                Thread.sleep((long) (60000.0f / sheet.getBpm()));

                for(Channel channel : channels)
                    channel.stop();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void destroy() {
        destroyChannels();
        destroyOpenAL();
    }
}
