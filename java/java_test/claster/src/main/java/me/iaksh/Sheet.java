package me.iaksh;

import java.util.ArrayList;

public abstract class Sheet {
    protected ArrayList<Section> sections;
    private int currentIndex;

    private void init() {
        sections = new ArrayList<>();
    }

    abstract void loadNotes();

    public Sheet() {
        init();
        loadNotes();
    }

    public void rewind() {
        currentIndex = 0;
    }

    public boolean eof() {
        return currentIndex >= sections.size();
    }

    public Section nextSection() {
        return sections.get(currentIndex++);
    }

    public Section currentSection() {
        if(currentIndex == sections.size())
            return sections.get(currentIndex - 1);
        else
            return sections.get(currentIndex);
    }
}
