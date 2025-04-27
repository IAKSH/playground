package me.iaksh;

public class Section {
    // 拍号
    private final int maxNotes;
    private final int standardNote;
    private int currentIndex;
    private Note[] notes;

    public Section(int maxNotes,int standardNote) {
        if(maxNotes < 0 || standardNote < 0)
            throw new IllegalArgumentException();
        this.maxNotes = maxNotes;
        this.standardNote = standardNote;
        notes = new Note[maxNotes];
        currentIndex = 0;
    }

    public void setNote(int i,Note note) {
        notes[i] = note;
    }

    public void rewind() {
        currentIndex = 0;
    }

    public Note nextNote() {
        return notes[currentIndex++];
    }

    public Note currentNote() {
        if(currentIndex == notes.length)
            return notes[currentIndex - 1];
        else
            return notes[currentIndex];
    }

    public boolean eof() {
        return currentIndex >= notes.length;
    }

    public int getMaxNotes() {
        return maxNotes;
    }

    public int getStandardNote() {
        return standardNote;
    }
}
