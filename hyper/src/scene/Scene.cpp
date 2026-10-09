#include <hyper/scene/Scene.h>

#include <hyper/App.h>

#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/TextureCache.h>

#include <donut/core/json.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unordered_map>

using namespace donut;

namespace hyper
{
    Scene::Scene(Context& ctx) : m_Ctx(ctx)
    {
        // Bindless on Vulkan (descriptor indexing is part of 1.2): ray tracing and
        // GPU-driven drawing both look materials and buffers up by index.
        if (ctx.api == nvrhi::GraphicsAPI::VULKAN || ctx.api == nvrhi::GraphicsAPI::D3D12)
        {
            nvrhi::BindlessLayoutDesc desc;
            desc.visibility = nvrhi::ShaderType::All;
            desc.maxCapacity = 16384;
            desc.registerSpaces = {
                nvrhi::BindingLayoutItem::RawBuffer_SRV(1),   // binding 0: index / vertex buffers
                nvrhi::BindingLayoutItem::Texture_SRV(2),     // binding 1: material textures
            };
            m_BindlessLayout = ctx.device->createBindlessLayout(desc);
            m_DescriptorTable = std::make_shared<engine::DescriptorTableManager>(ctx.device, m_BindlessLayout);
        }
        m_TextureCache = std::make_shared<engine::TextureCache>(ctx.device, std::make_shared<vfs::NativeFileSystem>(), m_DescriptorTable);
    }

    Scene::~Scene()
    {
        DoneLoading();
        if (m_Loader && m_Loader->joinable())
            m_Loader->join();
    }

    void Scene::Load(const std::filesystem::path& file)
    {
        if (m_Loader && m_Loader->joinable())
            m_Loader->join();

        m_Scene.reset();
        m_Sun.reset();
        m_File = file;
        m_TextureCache->Reset();
        m_TexturesFinished = false;
        m_Streaming.clear();
        m_ReadyUpdates = 0;
        m_LoaderDone = false;
        if (!m_Counted)
        {
            ++m_Ctx.loading;
            m_Counted = true;
        }
        m_State = State::Loading;
        log::info("hyper: loading scene %s", file.string().c_str());

        auto scene = std::make_shared<engine::Scene>(m_Ctx.device, *m_Ctx.shaders,
            std::make_shared<vfs::NativeFileSystem>(), m_TextureCache, m_DescriptorTable, nullptr);

        auto work = [this, scene, file]
        {
            // Only the loader thread touches m_Pending until m_LoaderDone; OnLoaded()
            // hands it to m_Scene on the main thread.
            m_LoaderOk = scene->Load(file);
            if (m_LoaderOk)
                m_Pending = scene;
            m_LoaderDone = true;
        };

        if (m_Ctx.capturing && !m_Ctx.asyncLoad)
        {
            work();
            OnLoaded();
            // Upload every texture now, so frame N of a capture never depends on timing. An
            // empty upload queue only means no decode has finished yet: wait for them all.
            while (m_TextureCache->GetNumberOfLoadedTextures() < m_TextureCache->GetNumberOfRequestedTextures())
                std::this_thread::yield();
            while (m_TextureCache->ProcessRenderingThreadCommands(*m_Ctx.common, 0.f)) { }
            m_TextureCache->LoadingFinished();
            m_TexturesFinished = true;
            // Donut's FinishedLoading (in OnLoaded) already wrote every material, before
            // these uploads: written again, with their textures.
            for (const auto& material : m_Scene->GetSceneGraph()->GetMaterials())
                material->dirty = true;
            m_Streaming.clear();
            DoneLoading();
        }
        else
        {
            m_Loader = std::make_unique<std::thread>(work);
        }
    }

    void Scene::DoneLoading()
    {
        if (m_Counted)
        {
            --m_Ctx.loading;
            m_Counted = false;
        }
    }

    void Scene::OnLoaded()
    {
        if (m_Loader && m_Loader->joinable())
            m_Loader->join();
        m_Loader.reset();

        m_Scene = std::move(m_Pending);
        if (!m_LoaderOk || !m_Scene)
        {
            log::error("hyper: scene failed to load");
            m_State = State::Failed;
            DoneLoading();
            return;
        }

        m_Scene->FinishedLoading(uint32_t(m_Ctx.frameIndex));
        ReadMaterialExtras();
        for (const auto& material : m_Scene->GetSceneGraph()->GetMaterials())
            m_Streaming.push_back({ material, 0u });

        for (const auto& light : m_Scene->GetSceneGraph()->GetLights())
        {
            if (light->GetLightType() == LightType_Directional)
            {
                m_Sun = std::static_pointer_cast<engine::DirectionalLight>(light);
                break;
            }
        }
        if (!m_Sun)
        {
            // Donut wants a leaf attached to the graph before it can be named or aimed.
            m_Sun = std::make_shared<engine::DirectionalLight>();
            m_Sun->angularSize = 0.53f;
            m_Sun->irradiance = 3.f;
            auto node = std::make_shared<engine::SceneGraphNode>();
            node->SetLeaf(m_Sun);
            m_Scene->GetSceneGraph()->Attach(m_Scene->GetSceneGraph()->GetRootNode(), node);
            m_Sun->SetName("Sun");
            m_Sun->SetDirection(dm::double3(0.3, -1.0, 0.25));
        }

        m_State = State::Ready;
        ++m_Generation;
        ++m_Revision;
        log::info("hyper: scene loaded (%zu mesh instances, %zu lights; textures: %u requested, %u decoded, %u uploaded)",
            m_Scene->GetSceneGraph()->GetMeshInstances().size(), m_Scene->GetSceneGraph()->GetLights().size(),
            m_TextureCache->GetNumberOfRequestedTextures(), m_TextureCache->GetNumberOfLoadedTextures(),
            m_TextureCache->GetNumberOfFinalizedTextures());
    }

    void Scene::Update(float deltaTime)
    {
        if (m_State == State::Loading && m_LoaderDone)
            OnLoaded();

        if (!m_TexturesFinished)
        {
            bool done = false;
            if (m_Ctx.capturing && m_Ctx.asyncLoad)
            {
                // A streaming capture (--async-load) takes the order a cold disk gives, the
                // one that hid the bugs, the same way every run: the scene first, one frame
                // drawn without its textures, then every texture at once.
                if (m_State == State::Ready && ++m_ReadyUpdates > 1)
                {
                    while (m_TextureCache->GetNumberOfLoadedTextures() < m_TextureCache->GetNumberOfRequestedTextures())
                        std::this_thread::yield();
                    while (m_TextureCache->ProcessRenderingThreadCommands(*m_Ctx.common, 0.f)) { }
                    done = true;
                }
            }
            else
            {
                // Donut counts a texture "loaded" once decoded (or failed) on a worker thread,
                // "finalized" once uploaded here. An empty upload queue only means no decode
                // finished since the last frame: streaming is over when every texture has
                // been decoded and nothing was left to upload.
                const bool decoded = m_TextureCache->GetNumberOfLoadedTextures() >= m_TextureCache->GetNumberOfRequestedTextures();
                done = !m_TextureCache->ProcessRenderingThreadCommands(*m_Ctx.common, 20.f) && decoded;
            }
            if (done && m_State == State::Ready)
            {
                m_TextureCache->LoadingFinished();
                m_TexturesFinished = true;
                DoneLoading();
            }
            if (m_State == State::Ready)
                RefreshStreamedMaterials();
        }

        if (m_State != State::Ready)
            return;

        if (animate)
        {
            animationTime += deltaTime;
            for (const auto& anim : m_Scene->GetSceneGraph()->GetAnimations())
            {
                const float duration = anim->GetDuration();
                (void)anim->Apply(duration > 0.f ? std::fmod(animationTime, duration) : 0.f);
            }
        }
        const bool changed = m_Scene->GetSceneGraph()->HasPendingStructureChanges();
        m_Scene->RefreshSceneGraph(uint32_t(m_Ctx.frameIndex));
        if (changed)
        {
            ++m_Revision;
            MapMaterialData();   // materials may have come or gone
        }
    }

    // Donut writes a material's constants (its textures' bindless indices among them) when
    // the material is dirty, and nothing marks it dirty when one of its textures arrives
    // later: streaming in, a material written early would miss it for good (cut-out leaves
    // sample alpha 0 and vanish). So each material still waiting is rewritten when one of
    // its own textures has been uploaded (Donut creates LoadedTexture::texture then).
    void Scene::RefreshStreamedMaterials()
    {
        std::erase_if(m_Streaming, [this](auto& entry)
        {
            engine::Material& m = *entry.first;
            uint32_t arrived = 0, total = 0;
            for (const auto* t : { &m.baseOrDiffuseTexture, &m.metalRoughOrSpecularTexture, &m.normalTexture,
                                   &m.emissiveTexture, &m.occlusionTexture, &m.transmissionTexture, &m.opacityTexture })
            {
                if (!*t)
                    continue;
                ++total;
                arrived += (*t)->texture ? 1u : 0u;
            }
            if (arrived != entry.second)
            {
                m.dirty = true;
                entry.second = arrived;
            }
            // Done when all are in, or when streaming is over (a texture that failed never arrives).
            return arrived == total || m_TexturesFinished;
        });
    }

    void Scene::RefreshBuffers(nvrhi::ICommandList* cmd)
    {
        if (m_State != State::Ready)
            return;
        m_Scene->RefreshBuffers(cmd, uint32_t(m_Ctx.frameIndex));
        if (m_MaterialDataDirty)
        {
            cmd->writeBuffer(m_MaterialDataBuffer, m_MaterialData.data(), m_MaterialData.size() * sizeof(uint32_t));
            m_MaterialDataDirty = false;
        }
    }

    // Donut drops glTF extras, so they're read from the file itself (a .gltf, or a .glb's
    // JSON chunk), and matched to Donut's materials by name.
    void Scene::ReadMaterialExtras()
    {
        m_Extras.clear();
        std::ifstream in(m_File, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.size() > 20 && std::memcmp(text.data(), "glTF", 4) == 0)
        {
            uint32_t length = 0;
            std::memcpy(&length, text.data() + 12, 4);
            text = text.substr(20, std::min<size_t>(length, text.size() - 20));
        }
        Json::Value root;
        std::string errors;
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (m_File.extension() != ".json" && reader->parse(text.data(), text.data() + text.size(), &root, &errors))
            for (const Json::Value& m : root["materials"])
                if (m.isMember("name") && m["extras"].isMember("hyper"))
                {
                    const Json::Value& h = m["extras"]["hyper"];
                    m_Extras[m["name"].asString()] = { h.get("data", 0u).asUInt(), h.get("rays", true).asBool() };
                }
        MapMaterialData();
        if (!m_Extras.empty())
            log::info("hyper: app data on %zu materials", m_Extras.size());
    }

    void Scene::MapMaterialData()
    {
        const auto& materials = m_Scene->GetSceneGraph()->GetMaterials();
        uint32_t count = 1;
        for (const auto& m : materials)
            count = std::max(count, uint32_t(m->materialID) + 1);
        m_MaterialData.assign(count, 0u);
        m_MaterialRays.assign(count, 1u);
        for (const auto& m : materials)
        {
            auto it = m_Extras.find(m->name);
            if (it == m_Extras.end())
                continue;
            m_MaterialData[m->materialID] = it->second.first;
            m_MaterialRays[m->materialID] = it->second.second ? 1u : 0u;
        }
        if (!m_MaterialDataBuffer || m_MaterialDataBuffer->getDesc().byteSize < count * sizeof(uint32_t))
            m_MaterialDataBuffer = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setByteSize(count * sizeof(uint32_t)).setStructStride(4)
                .setDebugName("hyper/material-data").setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_MaterialDataDirty = true;
    }

    bool Scene::IsStatic() const
    {
        if (m_State != State::Ready || !m_TexturesFinished)
            return false;
        return !animate || m_Scene->GetSceneGraph()->GetAnimations().empty();
    }

    float Scene::TextureProgress() const
    {
        const uint32_t requested = m_TextureCache->GetNumberOfRequestedTextures();
        return requested ? float(m_TextureCache->GetNumberOfLoadedTextures()) / float(requested) : 1.f;
    }

    nvrhi::IBindingLayout* Scene::BindlessLayout() const
    {
        return m_BindlessLayout;
    }

    nvrhi::IDescriptorTable* Scene::DescriptorTable() const
    {
        return m_DescriptorTable ? m_DescriptorTable->GetDescriptorTable() : nullptr;
    }

    void Scene::SetSunDirection(const dm::double3& direction)
    {
        if (!m_Sun || dm::length(m_Sun->GetDirection() - direction) < 1e-6)
            return;
        m_Sun->SetDirection(direction);
        if (m_Scene)
            m_Scene->GetSceneGraph()->Refresh(uint32_t(m_Ctx.frameIndex));
    }

    std::shared_ptr<engine::SceneGraph> Scene::Graph() const
    {
        return m_Scene ? m_Scene->GetSceneGraph() : nullptr;
    }

    dm::box3 Scene::Bounds() const
    {
        return m_Scene ? m_Scene->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox() : dm::box3::empty();
    }

    bool Scene::UseSceneCamera(Camera& camera) const
    {
        if (!m_Scene || m_Scene->GetSceneGraph()->GetCameras().empty())
            return false;
        const auto& sceneCamera = m_Scene->GetSceneGraph()->GetCameras().front();
        // View space looks down +Z (Donut flips glTF's -Z).
        const dm::affine3 viewToWorld = sceneCamera->GetViewToWorldMatrix();
        const dm::float3 position = viewToWorld.m_translation;
        camera.fly.LookAt(position, position + viewToWorld.m_linear.row2);
        if (auto perspective = std::dynamic_pointer_cast<engine::PerspectiveCamera>(sceneCamera))
            camera.verticalFov = perspective->verticalFov;
        const float radius = length(Bounds().diagonal()) * 0.5f;
        camera.fly.SetMoveSpeed(std::max(radius * 0.3f, 1.f));
        return true;
    }

    void Scene::FrameCamera(Camera& camera) const
    {
        const dm::box3 b = Bounds();
        if (b.isempty())
            return;
        const dm::float3 centre = b.center();
        const float radius = length(b.diagonal()) * 0.5f;
        const float distance = radius / std::sin(camera.verticalFov * 0.5f);
        const dm::float3 dir = normalize(dm::float3(-1.f, 0.35f, -0.6f));
        camera.fly.LookAt(centre - dir * distance * 1.1f, centre);
        camera.fly.SetMoveSpeed(std::max(radius * 0.3f, 1.f));
    }
}
