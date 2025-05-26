#pragma once

#include <functional>
#include <string>
#include <memory>
#include <map>

#include "model_interface.hpp"
#include "kinematic_cartesian.hpp"
#include "cartesian_augmented.hpp"

namespace GPMPC
{

namespace Model
{

using ModelCreator = std::function<std::unique_ptr<ModelInterface>()>;

class ModelLists
{
    private:
        std::map<std::string, ModelCreator> creators;
    
    public:
        ModelLists()
        {
            creators["Kinematic Cartesian"] = []() {
                return std::make_unique<KinematicCartesianModel>();
            };
            // creators["Cartesian Augmented"] = []() {
            //     return std::make_unique<CartesianAugmentedModel>();
            // };
        }

        std::unique_ptr<ModelInterface> create_model(const std::string &name)
        {
            auto it = creators.find(name);
            if (it != creators.end())
            {
                return it->second();
            }
            return nullptr;
        }
};

} // end namespace GPMPC

} // end namespace Model