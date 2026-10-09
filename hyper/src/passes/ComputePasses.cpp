#include <hyper/passes/ComputePasses.h>

#include <hyper/App.h>
#include <hyper/scene/Scene.h>

#include <hyper/shared.h>

#include <donut/engine/View.h>
#include <donut/shaders/view_cb.h>
#include <nvrhi/utils.h>

#include <cassert>
#include <cstring>

using namespace donut;

namespace hyper
{
    ComputePasses::ComputePasses(Context& ctx, std::vector<nvrhi::BindingLayoutItem> appItems,
                                 std::function<void(nvrhi::BindingSetDesc&)> appSet, const RtScene* rt)
        : m_Ctx(ctx), m_Rt(rt), m_AppSet(std::move(appSet))
    {
        m_View = ctx.device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "hyper/pass-view", 16));
        m_PrevView = ctx.device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "hyper/pass-view-prev", 16));
        m_Dummy = ctx.device->createTexture(nvrhi::TextureDesc().setDebugName("hyper/pass-dummy-uav").setWidth(1).setHeight(1)
            .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true).setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));

        nvrhi::BindingLayoutDesc desc;
        desc.setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(1))
            .addItem(nvrhi::BindingLayoutItem::PushConstants(3, c_PushBytes))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(1));
        for (const auto& item : appItems)
            desc.addItem(item);
        for (uint32_t i = 0; i < c_Srvs; ++i)
            desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(i));
        for (uint32_t i = 0; i < c_Uavs; ++i)
            desc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(i));
        if (m_Rt)
            RtScene::AddLayoutItems(desc);
        m_Layout = ctx.device->createBindingLayout(desc);
    }

    bool ComputePasses::Pipeline(nvrhi::ComputePipelineHandle& pipeline, const char* shader, const char* entry,
                                 nvrhi::IBindingLayout* bindless) const
    {
        auto cs = m_Ctx.shaders->CreateShader(shader, entry, nullptr, nvrhi::ShaderType::Compute);
        if (!cs)
            return false;
        nvrhi::ComputePipelineDesc desc;
        desc.setComputeShader(cs).addBindingLayout(m_Layout);
        if (m_Rt)
            desc.addBindingLayout(bindless);
        nvrhi::ComputePipelineHandle created = m_Ctx.device->createComputePipeline(desc);
        if (!created)
            return false;
        pipeline = created;
        return true;
    }

    void ComputePasses::BeginFrame(nvrhi::ICommandList* cmd, const engine::PlanarView& view, const engine::PlanarView& previous,
                                   Scene* scene)
    {
        m_Scene = scene;
        PlanarViewConstants v, p;
        view.FillPlanarViewConstants(v);
        previous.FillPlanarViewConstants(p);
        cmd->writeBuffer(m_View, &v, sizeof(v));
        cmd->writeBuffer(m_PrevView, &p, sizeof(p));
    }

    void ComputePasses::Dispatch(nvrhi::ICommandList* cmd, nvrhi::IComputePipeline* pipeline,
                                 std::initializer_list<nvrhi::ITexture*> srvs, std::initializer_list<nvrhi::ITexture*> uavs,
                                 uint32_t width, uint32_t height, const void* push, uint32_t pushBytes, uint32_t group)
    {
        if (!pipeline)
            return;
        assert(srvs.size() <= c_Srvs && uavs.size() <= c_Uavs && pushBytes <= c_PushBytes);
        nvrhi::BindingSetDesc d;
        d.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_View))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_PrevView))
            .addItem(nvrhi::BindingSetItem::PushConstants(3, c_PushBytes))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Sampler(1, m_Ctx.common->m_LinearWrapSampler));
        if (m_AppSet)
            m_AppSet(d);
        uint32_t slot = 0;
        for (nvrhi::ITexture* t : srvs)
            d.addItem(nvrhi::BindingSetItem::Texture_SRV(slot++, t ? t : m_Ctx.common->m_BlackTexture.Get()));
        for (; slot < c_Srvs; ++slot)
            d.addItem(nvrhi::BindingSetItem::Texture_SRV(slot, m_Ctx.common->m_BlackTexture));
        slot = 0;
        for (nvrhi::ITexture* t : uavs)
            d.addItem(nvrhi::BindingSetItem::Texture_UAV(slot++, t ? t : m_Dummy.Get()));
        for (; slot < c_Uavs; ++slot)
            d.addItem(nvrhi::BindingSetItem::Texture_UAV(slot, m_Dummy));
        nvrhi::ComputeState state;
        state.setPipeline(pipeline);
        if (m_Rt)
        {
            assert(m_Scene && "ComputePasses with an RtScene: give BeginFrame the scene");
            m_Rt->AddSetItems(d, *m_Scene);
        }
        nvrhi::IBindingSet* set = m_Sets.Get(m_Ctx.device, d, m_Layout);
        state.addBindingSet(set);
        if (m_Rt)
            state.addBindingSet(m_Scene->DescriptorTable());
        // NVRHI places barriers only when the bound sets change, so two dispatches with the
        // same inputs and outputs (one set from the cache) would get no UAV barrier between
        // them: the second could read what the first is still writing. Ask for the set's
        // states every time; a UAV already in that state gets its barrier.
        cmd->setResourceStatesForBindingSet(set);
        cmd->setComputeState(state);
        uint8_t constants[c_PushBytes] = {};
        if (push && pushBytes)
            std::memcpy(constants, push, pushBytes);
        cmd->setPushConstants(constants, c_PushBytes);
        cmd->dispatch((width + group - 1) / group, (height + group - 1) / group);
    }
}
