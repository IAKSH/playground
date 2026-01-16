#pragma once
#include <QWidget>
#include <opencv2/opencv.hpp>

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
    void on_auto_boundary_button_clicked();
    void on_threshold_slider_changed();
    void on_threshold_slider_released();
    void on_min_area_slider_changed();
    void on_min_area_slider_released();
private:
    Ui::ImageSelectionWidget *ui;
    cv::Mat pic,thresholded,denoised,colored;
    cv::Mat threshold(int threshold);
    cv::Mat remove_noise(const cv::Mat &binary, int min_area);
    cv::Rect get_min_rect(cv::Mat bin_ary);
    void process_pic();
    void apply_thresholded_pic();
    void apply_min_boundary();
};