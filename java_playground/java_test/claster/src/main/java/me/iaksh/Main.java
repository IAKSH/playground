package me.iaksh;

import org.lwjgl.openal.AL11;

public class Main {
    public static void main(String[] args) {
        Mixer m = new Mixer(120);

        Sheet[] sheets = new Sheet[4];
        sheets[0] = new Sheet() {
            @Override
            void loadNotes() {
                Section section = new Section(6,4);
                section.setNote(0,new Note(0.125f,0,0,6,0.05f));
                section.setNote(1,new Note(0.125f,0,0,5,0.05f));
                section.setNote(2,new Note(0.125f,0,0,3,0.05f));
                section.setNote(3,new Note(0.125f,0,0,5,0.05f));
                section.setNote(4,new Note(0.125f,0,0,3,0.05f));
                section.setNote(5,new Note(0.125f,0,0,2,0.05f));

                sections.add(section);
            }
        };
        sheets[1] = new Sheet() {
            @Override
            void loadNotes() {
                Section section = new Section(6,4);
                section.setNote(0,new Note(0.125f,1,0,5,0.01f));
                section.setNote(1,new Note(0.125f,1,0,3,0.01f));
                section.setNote(2,new Note(0.125f,1,0,5,0.01f));
                section.setNote(3,new Note(0.125f,1,0,3,0.01f));
                section.setNote(4,new Note(0.125f,1,0,2,0.01f));
                section.setNote(5,new Note(0.125f,1,0,3,0.01f));

                sections.add(section);
            }
        };
        sheets[2] = new Sheet() {
            @Override
            void loadNotes() {
                Section section = new Section(6,4);
                section.setNote(0,new Note(0.125f,-1,0,6,0.1f));
                section.setNote(1,new Note(0.125f,-1,0,5,0.1f));
                section.setNote(2,new Note(0.125f,-1,0,3,0.1f));
                section.setNote(3,new Note(0.125f,-1,0,5,0.1f));
                section.setNote(4,new Note(0.125f,-1,0,3,0.1f));
                section.setNote(5,new Note(0.125f,-1,0,2,0.1f));

                sections.add(section);
            }
        };
        sheets[3] = new Sheet() {
            @Override
            void loadNotes() {
                Section section = new Section(6,4);
                section.setNote(0,new Note(0.125f,1,0,5,0.025f));
                section.setNote(1,new Note(0.125f,0,0,2,0.025f));
                section.setNote(2,new Note(0.125f,1,0,0,0.025f));
                section.setNote(3,new Note(0.125f,0,0,3,0.025f));
                section.setNote(4,new Note(0.125f,1,0,5,0.025f));
                section.setNote(5,new Note(0.125f,0,0,0,0.025f));

                sections.add(section);
            }
        };

        m.play(sheets);
        m.destroy();
    }
}