
#ifndef DRIVER_AHT20_BASIC_H
#define DRIVER_AHT20_BASIC_H

#include "aht20_interface.h"

#ifdef __cplusplus
extern "C"{
#endif

/**
 * @defgroup aht20_example_driver aht20 example driver function
 * @brief    aht20 example driver modules
 * @ingroup  aht20_driver
 * @{
 */

/**
 * @brief  basic example init
 * @return status code
 *         - 0 success
 *         - 1 init failed
 * @note   none
 */
uint8_t aht20_basic_init(void);

/**
 * @brief  basic example deinit
 * @return status code
 *         - 0 success
 *         - 1 deinit failed
 * @note   none
 */
uint8_t aht20_basic_deinit(void);

/**
 * @brief      basic example read
 * @param[out] *temperature pointer to a converted temperature buffer
 * @param[out] *humidity pointer to a converted humidity buffer
 * @return     status code
 *             - 0 success
 *             - 1 read failed
 * @note       none
 */
uint8_t aht20_basic_read(float *temperature, uint8_t *humidity);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif