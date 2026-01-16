#pragma once
#include <QWidget>
#include <stampit/image.hpp>

namespace Ui {
    class ImageSelectionWidget;
}

class ImageSelectionWidget : public QWidget {
    Q_OBJECT
public:
    explicit ImageSelectionWidget(QWidget *parent = nullptr);
    ~ImageSelectionWidget();
    void on_load_button_clicked();
    void on_save_button_clicked();
    void on_threshold_slider_changed();
    void on_threshold_slider_released();
    void on_min_area_slider_changed();
    void on_min_area_slider_released();
private:
    Ui::ImageSelectionWidget *ui;
    stampit::Image original_img,final_img;
    void process_img(stampit::Image& img);
    void apply_thresholded_img(stampit::Image& img);
    void apply_final_img(stampit::Image& img);
    void apply_auto_boundary(stampit::Rect& rect);
};