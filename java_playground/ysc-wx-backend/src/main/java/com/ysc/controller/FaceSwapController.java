package com.ysc.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping("/face-swap")
public class FaceSwapController {

    @Value("${remaker.api.url}")
    private String remakerApiUrl;

    @Value("${remaker.api.key}")
    private String remakerApiKey;

    @PostMapping("/")
    public ResponseEntity<?> createAndCheckFaceSwapJob(@RequestParam("targetImageUrl") String targetImageUrl,
                                                       @RequestParam("swapImageUrl") String swapImageUrl) {
        try {
            // download images from URLs
            byte[] targetImage = downloadImage(targetImageUrl);
            byte[] swapImage = downloadImage(swapImageUrl);

            // create multipart request
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("target_image", new ByteArrayResource(targetImage) {
                @Override
                public String getFilename() {
                    return "target.jpg";
                }
            });
            body.add("swap_image", new ByteArrayResource(swapImage) {
                @Override
                public String getFilename() {
                    return "swap.jpg";
                }
            });

            HttpHeaders headers = new HttpHeaders();
            headers.set("accept", "application/json");
            headers.set("Authorization", remakerApiKey);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            RestTemplate restTemplate = new RestTemplate();
            URI uri = UriComponentsBuilder.fromHttpUrl(remakerApiUrl + "/face-swap/create-job").build().toUri();
            ResponseEntity<Map> createResponse = restTemplate.postForEntity(uri, requestEntity, Map.class);

            if (createResponse.getStatusCode() == HttpStatus.OK) {
                Map<String, Object> responseBody = createResponse.getBody();
                if (responseBody != null && responseBody.get("result") != null) {
                    Map<String, Object> result = (Map<String, Object>) responseBody.get("result");
                    String jobId = (String) result.get("job_id");

                    // call status API with polling
                    String statusUrl = remakerApiUrl + "/face-swap/" + jobId;
                    HttpHeaders statusHeaders = new HttpHeaders();
                    statusHeaders.set("accept", "application/json");
                    statusHeaders.set("Authorization", remakerApiKey);
                    HttpEntity<String> statusEntity = new HttpEntity<>(statusHeaders);

                    ResponseEntity<Map> statusResponse;
                    int attempts = 0;
                    boolean isCompleted = false;

                    while (attempts < 30 && !isCompleted) {
                        statusResponse = restTemplate.exchange(statusUrl, HttpMethod.GET, statusEntity, Map.class);
                        if (statusResponse.getStatusCode() == HttpStatus.OK) {
                            Map<String, Object> statusBody = statusResponse.getBody();
                            if (statusBody != null && statusBody.containsKey("code") && (int) statusBody.get("code") != 300102) {
                                return new ResponseEntity<>(statusBody, HttpStatus.OK);
                            }
                        }
                        attempts++;
                        TimeUnit.SECONDS.sleep(1); // Wait for 1 second before retrying
                    }
                    return new ResponseEntity<>("Face swap job timed out.", HttpStatus.REQUEST_TIMEOUT);
                } else {
                    return new ResponseEntity<>("Failed to retrieve job_id from create-job response.", HttpStatus.INTERNAL_SERVER_ERROR);
                }
            } else {
                return new ResponseEntity<>(createResponse.getBody(), createResponse.getStatusCode());
            }
        } catch (IOException | InterruptedException e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    private byte[] downloadImage(String imageUrl) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        try (InputStream inputStream = new URL(imageUrl).openStream()) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer, 0, buffer.length)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }
        return outputStream.toByteArray();
    }
}
