import java.security.InvalidParameterException;

public class ComputAction {
    private final int value;
    private final ComputeOption option;

    private ComputeOption castCharToOption(char optionChar) {
        switch(optionChar) {
            case '+' -> {return ComputeOption.Add;}
            case '-' -> {return ComputeOption.Subtract;}
            case '*' -> {return ComputeOption.Multiply;}
            case '\\' -> {return ComputeOption.Division;}
            default -> {throw new InvalidParameterException(String.format("unknown optionChar: %c",optionChar));}
        }
    }

    public ComputAction(int value,char option) {
        this.value = value;
        this.option = castCharToOption(option);
    }

    public ComputAction(int value,ComputeOption option) {
        this.value = value;
        this.option = option;
    }

    int executeWithBaseNum(int baseNum) {
        switch (option) {
            case Add -> {return baseNum + value;}
            case Subtract -> {return baseNum - value;}
            case Multiply -> {return baseNum * value;}
            case Division -> {return baseNum / value;}
            default -> {throw new RuntimeException();}
        }
    }
}
