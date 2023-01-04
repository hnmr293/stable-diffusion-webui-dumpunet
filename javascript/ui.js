onUiUpdate(() => {
    if (!globalThis.DumpUnet) {
        globalThis.DumpUnet = {};
    }
    const DumpUnet = globalThis.DumpUnet;

    DumpUnet.featureEnablerCallback = function () {
        if (DumpUnet.featureEnablerCallbackCalled) return;

        const app = gradioApp();
        if (!app || app === document) return;

        const unet_tab = app.querySelector('#dumpunet-features-tab'),
            prompt_tab = app.querySelector('#dumpunet-layerprompt-tab');
        if (!unet_tab || !prompt_tab) return;

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

        unet = apply(unet_tab, '#dumpunet-features-checkbox input[type=checkbox]');
        prompt = apply(prompt_tab, '#dumpunet-layerprompt-checkbox input[type=checkbox]');

        unet_tab.addEventListener('change', unet, false);
        prompt_tab.addEventListener('change', prompt, false);

        unet();
        prompt();

        DumpUnet.featureEnablerCallbackCalled = true;
    };

    onUiUpdate(DumpUnet.featureEnablerCallback);
});
