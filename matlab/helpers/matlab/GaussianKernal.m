function gKer = GaussianKernal(n_sigma, gaussian_sigma)
    kernel_size = round(n_sigma * gaussian_sigma); %R you changing it?
    x = -kernel_size:kernel_size;
    gaussian_kernel = exp(-(x.^2) / (2 * gaussian_sigma^2));
    gKer = gaussian_kernel / sum(gaussian_kernel);
end

