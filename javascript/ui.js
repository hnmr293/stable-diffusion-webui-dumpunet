onUiUpdate(() => {
    if (!globalThis.DumpUnet) {
        globalThis.DumpUnet = {};
    }
    const DumpUnet = globalThis.DumpUnet;

    DumpUnet.featureEnablerCallback = function () {
        if (DumpUnet.featureEnablerCallbackCalled) return;

        const app = gradioApp();
        if (!app || app === document) return;

        const tabs = {};
        for (let mode of ['txt2img', 'img2img']) {
            const unet_tab = app.querySelector(`#dumpunet-${mode}-features-tab`),
                prompt_tab = app.querySelector(`#dumpunet-${mode}-layerprompt-tab`);
            if (!unet_tab || !prompt_tab) return;
            tabs[mode] = { unet: unet_tab, prompt: prompt_tab };
        }

        const disableChildren = function (ele, excepts) {
            for (let e of ele.querySelectorAll('textarea, input, select')) {
                if (!excepts.includes(e))
                    e.disabled = true;
            }
        };

        const enableChildren = function (ele, excepts) {
            for (let e of ele.querySelectorAll('textarea, input, select')) {
                if (!excepts.includes(e))
                    e.disabled = false;
            }
        };

        const apply = function (parent, selector) {
            return () => {
                const cb = app.querySelector(selector)
                if (cb.checked) {
                    enableChildren(parent, [cb]);
                } else {
                    disableChildren(parent, [cb]);
                }
            };
        };

        for (let mode of ['txt2img', 'img2img']) {
            const tab = tabs[mode];
            const prompt = apply(tab.prompt, `#dumpunet-${mode}-layerprompt-checkbox input[type=checkbox]`);
            tab.prompt.addEventListener('change', prompt, false);
            prompt();
        }

        DumpUnet.featureEnablerCallbackCalled = true;
    };

    onUiUpdate(DumpUnet.featureEnablerCallback);
});
