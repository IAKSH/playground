#include <iostream>
#include <random>

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1,100);

    int num = distrib(gen);
    int input;

    std::cout << "Guess my number! (0~100)\n";

    while(true)
    {
        std::cin >> input;
        if(input == num)
            break;
        else if (input > num)
            std::cout << "Too big!\n";
        else
            std::cout << "Too small!\n";
    }

    std::cout << "Ok, you won.\n";
    return 0;
}
