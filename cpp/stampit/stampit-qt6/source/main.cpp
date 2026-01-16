#include <iostream>
#include <QApplication>
#include "image_selection_widget.hpp"

int main(int argc, char *argv[]) {
    std::cout << "Hello world!\n";
    QApplication app(argc, argv);
    ImageSelectionWidget w;
    w.show();
    return app.exec();
}
