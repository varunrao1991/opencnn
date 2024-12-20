#pragma once

#include "layer.h"

#include <memory>
#include <string>
#include <vector>

std::unique_ptr<Layer> CreateLayer(
    const std::string &, const std::string &, std::vector<std::string> &, std::shared_ptr<OpenclWrapper>);