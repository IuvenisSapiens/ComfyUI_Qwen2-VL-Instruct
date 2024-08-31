import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfyui_Qwen2-VL-Instruct.MultiplePathsInput",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("Comfyui_Qwen2-VL-Instruct")) {
            return;
        }
        switch (nodeData.name) {
            case "MultiplePathsInput":
                nodeType.prototype.onNodeCreated = function () {
                    this._type = "PATH";
                    this.inputs_offset = nodeData.name.includes("selective") ? 1 : 0;
                    this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        const target_number_of_inputs = this.widgets.find(
                            (w) => w.name === "inputcount"
                        )["value"];
                        if (target_number_of_inputs === this.inputs.length) return; // already set, do nothing

                        if (target_number_of_inputs < this.inputs.length) {
                            for (
                                let i = this.inputs.length;
                                i >= this.inputs_offset + target_number_of_inputs;
                                i--
                            )
                                this.removeInput(i);
                        } else {
                            for (
                                let i = this.inputs.length + 1 - this.inputs_offset;
                                i <= target_number_of_inputs;
                                ++i
                            )
                                this.addInput(`path_${i}`, this._type);
                        }
                    });
                };
                break;
        }
    },
    async setup() {
        const originalComputeVisibleNodes =
            LGraphCanvas.prototype.computeVisibleNodes;
        LGraphCanvas.prototype.computeVisibleNodes = function () {
            const visibleNodesSet = new Set(
                originalComputeVisibleNodes.apply(this, arguments)
            );
            for (const node of this.graph._nodes) {
                if (
                    (node.type === "SetNode" || node.type === "GetNode") &&
                    node.drawConnection
                ) {
                    visibleNodesSet.add(node);
                }
            }
            return Array.from(visibleNodesSet);
        };
    },
});
